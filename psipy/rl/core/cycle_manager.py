"""The :class:`CycleManager` manages communication and holds cycle statistics.

The :class:`CycleManager` should never be instantiated directly, instead one
should make use of the :const:`CM` variable, referencing the global single
:class:`CycleManager` instance aka singleton.

Communication takes place in non-blocking fashion through zmq pub/sub sockets.

Currently published messages are NOT conflated (which would mean that only the
latest message is kept available for collection by subscribers). Subscribers not
receiving messages as fast as they are published might lag behind the loop,
precautions need to be taken on the subscriber's side to mitigate this!

Messages are published on a single socket into the following topics.

- ``lifecycle``: Data contains at least a ``event`` key with of these values:

  - ``episode_starts``: Called on episode starts, contains additional
    information about the upcoming loop.
  - ``episode_stops``: Current episode ended.
  - ``exit``: Loop process exit, should trigger terminal ui shutdown.
    **Currently not implemented**.

- ``step``:

  - ``state``: ``values``, ``terminal``, ``cost`` and ``meta``
  - ``action``: Current transition's action's values
  - ``stats``: Timer statisics
  - ``step``: Step counters current value

- ``exception``: Exception caught on the loop's top level including traceback.
- ``logs``: :mod:`logging` logs.
- ``custom``: Task specific information.

    - ``tab``: the name of the :class:`~psipy.rl.io.terminal_interface.Tab`
               to load for the custom tab
    - ``info``: a dictionary of information that the
                :class:`~psipy.rl.io.terminal_interface.Tab` deals with

Similarly, the :class:`CycleManager` subscribes to a :class:`zmq.Socket` on a
different port, expecting the following topics.

- ``cmd``: Commands to be applied to the loop's process. Currently the following
  commands are supported:

  - ``episode_stop``: Stop the current episode, maybe starting a new one when
    some are left on the ``run`` stack.
  - ``exit``: Exit loop process.

- ``plant``: Pass keyword arguments to the currently active plant through its
             :meth:`~psipy.rl.plant.plant.user_input` method.
- ``control``: Pass keyword arguments to the currently active controller through
               its :meth:`~psipy.rl.control.controller.user_input` method.
               NOTE: Currently not implemented.

A component interfacing with the :class:`CycleManager`'s :class:`zmq.Socket`
instances is the :class:`psipy.rl.io.terminal_interface.TerminalInterface`.

"""

import _thread
import json
import logging
import time
import traceback
from functools import partial
from math import sqrt
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional, Tuple, Union

import zmq
from zmq.error import ZMQError
from zmq.log.handlers import PUBHandler

from psipy.core.io.json import NativelyJSONEncodable, json_encode
from psipy.core.io.logging import LOG_LEVEL_STATUS_NUM
from psipy.core.threading_utils import StoppableThread

__all__ = ["CM"]


LOG = logging.getLogger(__name__)


def setfield(obj: Dict, key: str, val: Any) -> None:
    """Sets a given key on a given dict to a given value inplace.

    Example::

        >>> obj = dict()
        >>> setfield(obj, "a", 1)
        >>> obj["a"]
        1

    """
    obj[key] = val


class Timer:
    """Timer keeping exponential moving statistics.

    The timer implementation is currently completely tailored to the use in
    :class:`CycleManager`, as it is only used there. If a less weird interface
    then passing in a callback function is required, this class will need to be
    refactored.

    Args:
        callback: Called given a (ema, emstd) after every update.

    Example:

        >>> stats = dict()
        >>> with Timer(lambda em: setfield(stats, "test", em)) as t:
        ...     time.sleep(1)
        >>> ema, emstd = stats["test"]
        >>> round(ema), emstd
        (1, 0.0)

    """

    #: ``2/(N+1)`` where N is the smoothing parameter. Larger N takes longer to
    #: 'converge', but the average it converges to will be more stable. See
    #: pictures on https://www.investopedia.com/terms/e/ema.asp for a general
    #: idea.
    _alpha = 2 / (100 + 1)

    def __init__(self, callback: Callable[[Tuple[float, float]], None]):
        self._callback = callback
        self._tick: Optional[float] = None
        self._initialized = False
        self._ema = 0.0
        self._emvar = 0.0
        self._emstd = 0.0
        self._latest = 0.0

    def __str__(self):
        """Converts the timer to a string."""
        return f"Timer EMA:{self._ema}|EMSTD:{self._emstd}"

    def __enter__(self):
        """Starts a cycle measurement."""
        self.tick()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops a cycle measurement, updating tracked statistics."""
        self.tock()

    def tick(self):
        """Starts a cycle measurement."""
        if self._tick is not None:
            raise ValueError("Timer already started!")
        self._tick = time.time()

    def tock(self):
        """Stops a cycle measurement, updating tracked statistics."""
        if self._tick is None:
            raise ValueError("Timer never started!")
        elapsed = time.time() - self._tick
        self._latest = elapsed
        self._tick = None
        if not self._initialized:
            self._ema = elapsed
            self._emstd = 0.0
            self._initialized = True
        delta = elapsed - self._ema
        self._ema = self._ema + self._alpha * delta
        self._emvar = (1 - self._alpha) * (self._emvar + self._alpha * delta ** 2)
        self._emstd = sqrt(self._emvar)
        self._callback((self._ema, self._emstd))

    def ticktock(self):
        """Stops the last (if any) and starts the next cycle."""
        try:
            self.tock()
        except ValueError:
            pass
        return self.tick()

    @property
    def time(self):
        """Gets the current exponential moving average value of the timer."""
        return self._ema


class PubSub(StoppableThread):
    """Handles message publishing in non-blocking fashion."""

    #: Contains incoming commands.
    cmds: Dict[str, bool]

    def __init__(self):
        super().__init__(daemon=True)
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.sub = self.ctx.socket(zmq.SUB)
        self.queue = Queue()
        self.cmds = dict()
        self.start()

    def _find_open_port(self):
        """Finds free random ports for both pub and sub."""
        kwargs = dict(min_port=49152, max_port=65536, max_tries=100)
        port = self.pub.bind_to_random_port("tcp://*", **kwargs)
        port_sub = self.sub.bind_to_random_port("tcp://*", **kwargs)
        return port, port_sub

    @staticmethod
    def _setup_zmq_logger(zmq_socket: zmq.Socket, level: str = "INFO") -> None:
        """Setup a logging handler publishing logs to a zmq topic."""
        handler = PUBHandler(zmq_socket)
        handler.setFormatter(
            logging.Formatter("%(levelname)-8s %(asctime)-8s %(message)s", "%H%M%S")
        )
        handler.root_topic = "logs"
        if hasattr(logging, "STATUS"):
            # Add formatter for psipy.core.io.logging.LOG_LEVEL_STATUS_NUM logs.
            fmt = handler.formatters[logging.INFO]
            handler.setFormatter(fmt, level=LOG_LEVEL_STATUS_NUM)
        logger = logging.getLogger()
        logger.setLevel(level)
        handler.setLevel(level)
        logger.addHandler(handler)

    def setup(self, port: int, log_level: str = "INFO") -> None:
        """Configure PubSub thread.

        Args:
            port: Publish to ``port``, subscribe to ``port + 1``.
            log_level: Logging level for logs published through ZMQ ``logs`` topic.
        """
        LOG.info("Connecting zmq pub/sub...")
        port_sub = port + 1
        try:
            self.pub.bind(f"tcp://*:{port}")
            self.sub.bind(f"tcp://*:{port_sub}")
        except ZMQError:
            LOG.error(f"CM failed to connect zmq to ports {port} and {port_sub}!")
            port, port_sub = self._find_open_port()
            LOG.warning(f"CM fell back to random zmq ports {port} and {port_sub}.")
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self._setup_zmq_logger(self.pub, log_level)
        LOG.info(f"Connected zmq to ports pub {port} and sub {port_sub}.")

    def send(self, topic: str, data: Dict[str, Any]):
        return self.queue.put_nowait((topic, data))

    def _receive(self):
        try:
            topic, msg = self.sub.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            return None
        topic = topic.decode()
        msg = msg.decode()
        if topic == "cmd":
            self.cmds["should_stop"] = msg in ("episode_stop", "exit")
            self.cmds["should_exit"] = msg == "exit"
        else:
            self.cmds[topic] = json.loads(msg)

    def _maybe_exit(self) -> None:
        """Exit main thread if requested so in the commands.

        This triggers a :class:`KeyboardInterrupt` in the main thread, which
        should by itself again be handled gracefully by the loop.
        """
        if self.cmds.get("should_exit", False):
            _thread.interrupt_main()

    def run(self):
        while not self.stopped():
            self._receive()
            self._maybe_exit()
            try:
                topic, data = self.queue.get(timeout=0.1)
            except Empty:
                pass
            else:
                self.pub.send_multipart(
                    msg_parts=[topic.encode(), json_encode(data).encode()]
                )

    def reset(self):
        if not self.queue.empty():
            # When in a multiprocessing setting, the queue may not empty completely,
            # causing a memory leak. This ensures the queue is empty when reset.
            self.queue = Queue()
        self.cmds = dict()


class CycleManager:
    """Manages runtime measures of the loop and individual sub-parts."""

    #: Timers managed by the cycle manager.
    _timers: Dict[str, Timer]

    #: Running average and standard deviation of timers.
    _stats: Dict[str, Tuple[float, float]]

    #: Whether CM has been fully setup for use.
    _is_setup: bool = False

    #: Some background/meta information, updated on every episode start.
    _meta: Dict[str, Union[str, int]]

    #: Episode cycle counter.
    _cycle: int

    #: Whether the current episode should stop right now.
    _should_stop: bool

    #: Timestamp of very first plant interaction in the current episode.
    _interaction_ts0: float

    #: Timestamp of most recent plant interaction.
    _interaction_ts: float

    #: Storage for whatever diagnostic information to be available globally.
    diagnostics: Dict[str, NativelyJSONEncodable]

    def __init__(self):
        """Initializes a new cycle manager, tracking of statistics in a shared state."""
        self.pubsub = PubSub()
        self._timers = dict()
        self._stats = dict()
        self._cycle = 0
        self._should_stop = False
        self.diagnostics = dict()

    def setup(self, port: Optional[int] = None, log_level: str = "INFO"):
        """Sets up the :class:`CycleManager`, should be called on loop startup.

        Args:
            port: Publish to ``port``, subscribe to ``port + 1``.
        """
        self.reset()
        self.pubsub.setup(port, log_level)
        self._is_setup = True
        return self

    def reset(self):
        """Resets the internal state.

        Required between episodes or useful when used in a stateful environment
        like a jupyter notebook.
        """
        self._timers.clear()
        self._stats.clear()
        self.pubsub.reset()
        self._should_stop = False
        self._cycle = 0
        self._interaction_ts0 = -1
        self._interaction_ts = -1
        self.diagnostics = dict()

    def __getitem__(self, name: str) -> Timer:
        """Gets a timer managed by the :class:`CycleManager`."""
        if name not in self._timers:
            callback = partial(setfield, self._stats, name)
            self._timers[name] = Timer(callback)
        return self._timers[name]

    @property
    def available_timers(self):
        """List all timers currently tracked by the :class:`CycleManager`."""
        return list(self._timers.keys())

    @property
    def cycle(self):
        """Current cycle count."""
        return self._cycle

    def notify_episode_starts(
        self, episode_number: int, plant: str, control: str
    ) -> None:
        """Resets the tracked stats to their initial state."""
        self.reset()
        self._meta = dict(episode=episode_number, control=control, plant=plant)
        package: Dict[str, Union[str, int]] = {**self._meta, "event": "episode_starts"}
        self.pubsub.send("lifecycle", package)

    def step(self, data: Dict[str, Any], increment_step_counter: bool = True):
        """Communicates with outside world, should be called once per cycle.

        Publishes current information received from loop (state, action and
        metadata) and additional information like step count and cycle stats
        while also listening for commands coming in, like episode stop or loop
        exit commands. Exits triggered by a user through the
        :class:`~psipy.rl.io.terminal_interface.TerminalInterface` enter the
        loop process in this method, result in the loop to both stop the current
        episode and finish itself, triggering the deletion of the
        :class:`CycleManager` (``__del__``), communicating the shutdown back to
        the :class:`~psipy.rl.io.terminal_interface.TerminalInterface`, allowing
        it to quit as well.
        """
        if increment_step_counter:
            self._cycle += 1
            CM.diagnostics["cycle"] = self.cycle
        with self["CM.step"]:
            data = {
                **data,
                "stats": self._stats,
                "interaction_time": self._interaction_ts - self._interaction_ts0,
                **self._meta,
            }
            self.pubsub.send("step", data)
            self._should_stop = self.pubsub.cmds.get("should_stop", False)

    def step_custom(
        self, tab: str = "blank", info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Publishes custom information to the TUI interface.

        Can be called multiple times per loop.
        """
        if info is None:
            info = dict()
        with self["CM.custstep"]:
            package = dict(tab=tab, info=info)
            self.pubsub.send("custom", package)

    def set_time(self, ts: float) -> None:
        """Sets the current plant time, used for tracking interaction time."""
        if self._interaction_ts0 == -1:
            self._interaction_ts0 = ts
        self._interaction_ts = ts

    def handle_exception(self, exception: Exception):
        """Handles exceptions by communicating them to the outside world."""
        msg = dict(exception=str(exception), traceback=traceback.format_exc())
        self.pubsub.send("exception", msg)

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop.

        May raise ``NotNotifiedOfEpisodeStart`` in the future, currently does not.
        """
        self._timers.clear()
        self.pubsub.send("lifecycle", dict(event="episode_stops"))

    def is_setup(self) -> bool:
        """Checks whether the :class:`CycleManager` has been setup properly."""
        return self._is_setup

    def should_stop(self, max_steps: int = -1) -> bool:
        """Checks whether the current episode should stop."""
        if max_steps > 0 and self.cycle >= max_steps:
            return True
        return self._should_stop


#: Global :class:`CycleManager` singleton. There can always just be a single
#: :class:`CycleManager` instance, which is this one. The :class:`CycleManager`
#: class should never be instantiated directly besides for creating this
#: instance right here.
CM: CycleManager = CycleManager()
