# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Terminal User Interface for Reinforcement Learning projects.

The TUI receives via zmq information from the RL :class:`Loop` and displays
it in a readable format. TUIs will always display the current plant and
controller, state, action, meta, and cycle times. Optionally, the TUI can
display information in another tab, internally called the "Custom Tab",
because the information displayed here is dependent on the customer. This
information can take any format, for example bar graphs or array representations.
All that needs to be done is implement a create of :class:`Tab` and implement its
:meth:`print`.

Information is sent via the Cycle Manager's (:class:`CycleManager`) :meth:`step` and
:meth:`step_custom` methods.  Step is called once and only once during each step
in the loop, while :meth:`step_custom` can be called multiple times within one loop.
All information sent over the custom step is conglomerated into one message that is
then printed in the custom tab.

Errors in the TUI close the TUI gracefully and print the stacktrace to the terminal.
Errors within the loop open a popup with error and stacktrace within the TUI, and must
be acknowledged by the user by hitting `Enter`. The loop can continue to run while this
error window is open. For an example error, hit `d` in the TUI. A generic error will
appear.

The TUI is implemented in Urwid. More information here:
http://urwid.org/reference/index.html
and and overview of what the different widgets do is here:
http://urwid.org/manual/widgets.html
"""
import logging
from abc import abstractmethod
from argparse import ArgumentParser
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import tabulate as tb
import urwid
import zmq
from urwid import AttrMap, Columns, Filler, Frame, LineBox, Overlay, Padding, Pile, Text

from psipy.core.io import json_decode
from psipy.core.threading_utils import StoppableThread
from psipy.rl.plant.plant import Numeric

__all__ = ["TerminalInterface"]
__version__ = "3.0"

LOG = logging.getLogger(__name__)

_number_dtypes = (
    np.generic,
    float,
    int,
)

NUM_LOGS = 5
HEIGHT_LOGS = NUM_LOGS + 2  # plus top and bottom lines
HEIGHT_HEADER = 5

palette = [
    ("highlight gray", "black", "light gray"),
    ("box gray", "light gray", "black"),
    ("underline", "light gray,underline", "black"),
    ("bold", "light gray,bold", "black"),
    ("error", "yellow", "dark red"),
    ("error title", "yellow,blink,bold", "dark red"),
    ("button select", "white,standout", "black"),
    ("graph normal", "brown", "black"),
    ("graph inverse", "black", "brown"),
    ("graph 100%", "light red", "black"),
]


class Tab:
    """A tab in the TUI interface.

    In each call to :meth:`print` from the TUI, the tab is updated with incoming
    information from ZMQ and any other information calculated within the tab. Tabs
    are always updated, and their visibility is determined by the TUI itself, not the
    tab. Custom tabs must:

        1. Build their urwid widget structure.
        2. Define an update function (:meth:`print`) which takes any number of arguments
           to alter the urwid text. See :class:`MainTab` for an example.

    By providing a header, footer, and/or logs box to a subclass, the parameters
    from the provider's boxes can be maintained across pages. In generally all cases,
    these parameters should be the boxes from the :class:`Main` :class:`Tab`. When
    doing so, it is not necessary to worry about printing to those boxes in
    :meth:`print`, since all tabs are always printed, and they will be updated in
    another tab's :meth:`print`.

    Every subclassed tab, besides :class:`Main` must be added to the
    :param:`TAB_REGISTER` so that they can be loaded via strings.

    Args:
        header: the header box
        footer: the footer box
        logs: the logs box
    """

    #: The final frame that contains all urwid widgets
    _frame: urwid.Frame

    def __init__(
        self,
        header: Optional[urwid.Widget] = None,
        footer: Optional[urwid.Widget] = None,
        logs: Optional[urwid.Widget] = None,
    ):
        self.header_box = header
        self.footer = footer
        self.log_box = logs

        self.create()

    @abstractmethod
    def create(self) -> None:
        """Create the entire frame.

        All components that need to be referenced from the outside should
        be self.* components!
        """
        raise NotImplementedError

    @abstractmethod
    def print(self, *args, **kwargs) -> None:
        """Updates the components within the tab."""
        raise NotImplementedError

    @property
    def frame(self):
        return self._frame


class TextButton(urwid.Button):
    button_left = urwid.Text("")
    button_right = urwid.Text("")


class Array:
    # TODO: Alignment on all columns since it jiggles
    def __init__(self, arr: Union[np.ndarray, List[List[Numeric]]]):
        self.arr = arr
        if isinstance(self.arr, np.ndarray):
            self.arr = self.arr.tolist()

    def _get_repr(self, labels: Optional[List[str]] = None) -> str:
        repr = ""
        rows = 0
        cols = 0
        for i, row in enumerate(self.arr):
            nums = ""
            for col in row:
                nums += f"{round(col,2)} "
            repr += f"{nums}"
            if labels:
                repr += f" -{labels[i]}"
            repr += "\n"
            rows += 1
            cols = max(cols, len(repr.split("\n")[-1]))

        return repr

    def create(self, labels: Optional[List[str]] = None) -> str:
        arr = self._get_repr(labels)
        return arr


def format_text(text: str, attr: str) -> Tuple[str, str]:
    """Pack text in a tuple according to urwid's requirements for attributes."""
    return (attr, text)


def format_date(date):
    return date.strftime("%H:%M:%S %Y-%m-%d")


def format_timer(ts: float) -> str:
    if ts < 100:
        return f"{ts:.2f}s"
    ds = datetime.fromtimestamp(ts)
    if ts < 3600:
        return ds.strftime("%M:%S.%f")[:-3]
    return ds.strftime("%H:%M:%S.%f")[:-3]


def print_dict(
    dictionary: Dict[str, Any], tab_size: int = 1, align_on_decimal: bool = False
) -> str:
    """Prints k,v of a dictionary line by line, aligning on decimals if desired."""
    string = ""
    tab_size = " " * tab_size

    # To align on decimal, all values must be numeric (string numbers don't count!)
    if align_on_decimal:
        if not all([isinstance(val, _number_dtypes) for val in dictionary.values()]):
            # Bad state return; check for non-numbers!
            align_on_decimal = False
    max_len_key = max([len(key) for key in dictionary])
    # Max displayable number length before decimals become unaligned, e.g. 12345.678
    max_len_value = 9

    for key, value in dictionary.items():
        value = f"{value:3.3f}" if align_on_decimal else value
        tab = " " * (max_len_key - len(str(key)))
        alignment = (
            " " * (max_len_value - len(value.split(".")[0])) if align_on_decimal else ""
        )
        string += f"{key}{tab}:{tab_size}{alignment}{value}\n"

    return string


def center_on_char(text: str, char: str = "|") -> str:
    """Centers string on a char. Must use urwid's centered text in conjunction."""
    # Get position of the centering character
    text_pipe = text.find(char) + 1  # +1 to shift to proper center
    # Get midpoint of the text
    text_mid = len(text) // 2
    # Determine if extra spaces should be added to the front or the back of the str
    text_diff = text_mid - text_pipe
    front_setup = np.sign(text_pipe - text_mid)
    # Add the spaces to center
    if front_setup:
        text = (" " * text_diff * 2) + text
    else:
        text = text + (" " * text_diff * 2)
    return text + "\n"


def create_box(
    text: Text,
    title: str,
    title_align: str = "center",
    title_attr: Union[str, Tuple] = "highlight gray",
    valign: str = "top",
    top_pad: int = 0,
    left_pad: int = 0,
    bottom_pad: int = 0,
    right_pad: int = 0,
):
    """Create a box in Urwid."""
    # Filler is up and down buffer
    filler = Filler(text, top=top_pad, bottom=bottom_pad, valign=valign)
    # Padding is side to side buffer
    padding = Padding(filler, left=left_pad, right=right_pad)
    return LineBox(padding, title=title, title_attr=title_attr, title_align=title_align)


def generate_footer(
    options: List[Tuple[str, str]], sep: str = "|"
) -> Tuple[urwid.Widget, urwid.Widget, urwid.Widget]:
    """Create footer with buttons for navigation.

    Args:
        options: list of tuples, where the tuples are two strings
                 where the first string is the text, and the second string
                 is the letter used to activate that text command.
    """
    text = [" "]  # initial space to not be against the border
    for i, option in enumerate(options):
        end = f" {sep} " if i != len(options) - 1 else ""
        text.append(f"{option[1]}: {option[0]}{end}")
    main_button = TextButton("Main")
    custom_button = TextButton("Agent")
    return Text(text), main_button, custom_button


class TUISubscriber(StoppableThread):
    """The subscriber manages incoming messages in its own thread.

    Keeps track of the latest message per topic, making them available to the
    terminal interface loop. It may happen that individual messages are skipped
    if the terminal interface loop does not read them from this Subscriber
    fast enough. For messages on the ``step`` channel this is on purpose (as the
    primary control loop may run faster than the terminal interface), for other
    topics (like ``exception`` or ``lifecycle``) the terminal interface has to
    be able to degrade accordingly.

    Also it is to note, that messages can only be read once! Once a message is
    read from the subscriber, it is deleted. This is implemented in order to not
    have an event happen twice in quick succession.
    """

    #: Latest conflated messages for individual topics.
    _data: Dict[str, str]

    #: Message buffers for individual topics.
    _buffers: Dict[str, List[Dict[str, str]]]

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port

        self._data = dict()
        self._buffers = dict(logs=[])

        self.zmq = zmq.Context()
        self.sub = self.zmq.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.sub.setsockopt(zmq.RCVTIMEO, 1000)  # wait 1 sec at most per iter
        self.sub.connect(f"tcp://localhost:{port}")

    def __del__(self):
        self.stop()
        self.sub.close()
        self.zmq.destroy()

    def __getitem__(self, topic: str) -> Optional[Dict[str, Any]]:
        if topic == "logs":
            if len(self._buffers[topic]) == 0:
                return None
            return self._buffers[topic].pop()
        msg = self._data.pop(topic, "null")  # json_decode("null") == None
        return cast(Optional[Dict[str, Any]], json_decode(msg))

    def run(self):
        while not self.stopped():
            try:
                temp = self.sub.recv_multipart()
            except zmq.ZMQError:
                continue
            try:
                topic, msg = temp
            except ValueError:  # workaround for handling messed up messages
                continue
            topic = topic.decode()
            if topic.startswith("logs."):
                topic, level = topic.split(".", maxsplit=1)
                self._buffers[topic].append(dict(level=level, msg=msg.decode()))
            else:
                self._data[topic] = msg.decode()


class MainTab(Tab):
    def __init__(self, start_time: datetime, ports: Tuple[int, int]):
        self.start_time = start_time
        self.ports = ports
        super().__init__()

    def create(self) -> None:
        # Header related objects
        self.header_text = Text("Initializing...", align="center")
        header_title = "PSIORI Loop Monitor v3.0 | i/o "
        self.header_box = create_box(
            self.header_text, header_title, "left", title_attr=""
        )

        # Status box related objects
        self.status_text = Text("Awaiting data...")
        status_title = "Status"
        self.status_box = create_box(self.status_text, status_title)

        # Meta box related objects
        self.meta_text = Text("Awaiting data...")
        meta_title = "Meta Information"
        self.meta_box = create_box(self.meta_text, meta_title)

        # Cycle manager related objects
        self.cm_text = Text("Awaiting cycle information...")
        cm_title = "Cycle Manager Info"
        self.cm_box = create_box(self.cm_text, cm_title)

        # Log box related objects
        self.log_text = Text("Awaiting logs...")
        log_title = "Logs"
        self.log_box = create_box(self.log_text, log_title)

        # Footer related objects
        # To add commands, add more tuples here and
        # their respective actions in the TUI class
        footer_text, self.main_button, self.custom_button = generate_footer(
            [("Quit TUI", "q"), ("Stop Episode", "s"), ("Exit loop", "e")]
        )
        mainbutton_color = AttrMap(self.main_button, "", "button select")
        custombutton_color = AttrMap(self.custom_button, "", "button select")
        self.footer = Columns(
            [
                footer_text,
                (10, Text("Tabs:")),
                (10, mainbutton_color),
                (10, custombutton_color),
            ]
        )

        # Create proper box layout
        right_boxes = Pile([self.meta_box, self.cm_box])
        center_boxes = Columns([self.status_box, right_boxes])
        main_body = Pile(
            [
                (HEIGHT_HEADER, self.header_box),
                center_boxes,
                (HEIGHT_LOGS, self.log_box),
            ]
        )

        # Assemble the widgets into the widget layout
        self._frame = AttrMap(Frame(body=main_body, footer=self.footer), "box gray")

    def print_header(self, plant_name, controller_name) -> None:
        """Returns the header text with the top line being bold."""
        setup = f"Plant: {plant_name} | Controller: {controller_name}"
        times = (
            f"Start Time: {format_date(self.start_time)} | "
            f"Current Time: {format_date(datetime.now())}"
        )
        elapsed = f"Elapsed: {str(datetime.now() - self.start_time)}"

        setup = center_on_char(setup)
        times = center_on_char(times)

        setup = format_text(setup, "bold")
        rest = f"{times}{elapsed}"

        self.header_text.set_text([setup, rest])
        self.header_box = cast(urwid.LineBox, self.header_box)
        self.header_box.set_title(
            f"PSIORI Loop Monitor v3.0 | i/o {self.ports[0]}/{self.ports[1]}"
        )

    def print_status(self, status_dict: Dict[str, Any]) -> None:
        """Returns the status text, a list due to styling."""
        # Create the State section
        state = format_text("State\n", "underline")
        string = "Values\n"
        string += print_dict(status_dict["values"], align_on_decimal=False)
        # Create the Return section
        string += "\nReturns\n"
        string += f"Cost: {status_dict['cost']}\n"
        string += f"Terminal: {status_dict['terminal']}\n"
        # Create the Action section
        action = format_text("\nActions\n", "underline")
        action_string = print_dict(status_dict["action"], align_on_decimal=False)

        self.status_text.set_text([state, string, action, action_string])

    def print_meta(self, meta_dict: Dict[str, Any]) -> None:
        """Returns the meta text, currently just a dict print."""
        self.meta_text.set_text(print_dict(meta_dict))

    def print_cm(self, cm_dict: Dict[str, Any]) -> None:
        """Returns the cycle times with stddev."""
        for name, timer in cm_dict.items():
            avg, std = timer
            avg *= 1000
            std *= 1000
            cm_dict[name] = f"{avg:7.2f}ms ±{std:5.2f}"
        self.cm_text.set_text(print_dict(cm_dict))

    def print(  # type:ignore
        self,
        plant_name: str,
        controller_name: str,
        status_dict: Dict[str, Any],
        meta_dict: Dict[str, Any],
        cm_dict: Dict[str, Any],
    ) -> None:
        self.print_header(plant_name, controller_name)
        self.print_status(status_dict)
        self.print_meta(meta_dict)
        self.print_cm(cm_dict)


class ScalpingTab(Tab):

    #: Horizontal line at bin full level
    lines = [100]

    def create(self) -> None:
        self.state_text = Text("Unused.", align="center")
        self.state_title = "Plant Summary"
        self.state_box = create_box(self.state_text, self.state_title, valign="middle")
        self.levels = urwid.BarGraph(
            ["graph normal", "graph inverse"], ["graph 100%", "graph inverse"]
        )
        self.levels_box = LineBox(
            self.levels, title="Bin Levels", title_attr="highlight gray"
        )
        # Create proper layout
        cust_cols = Columns([self.state_box, self.levels_box])
        cust_body = Pile(
            [(HEIGHT_HEADER, self.header_box), cust_cols, (HEIGHT_LOGS, self.log_box)]
        )
        self._frame = AttrMap(Frame(body=cust_body, footer=self.footer), "box gray")

    def _convert_to_tuples(self, levels: List[Numeric]) -> List[Tuple[Numeric, ...]]:
        """Bar graphs expect list of tuples for bars; this converts to that format"""
        return [(num,) for num in levels]

    def _input(self, state: List[List[Numeric]]) -> str:
        state = Array(state)
        return state.create(["Row 1", "Row 2"])

    def _metric(self, state: Dict[str, Any]) -> str:
        """Add metric with multiple values.

        Args:
            state: Metric with multiple stats, i.e.

        .. code-block:: json

            {
                "name": "Fills",
                "metrics": {
                    "Mean": np.mean(fills[cluster]),
                    "Median": np.median(fills[cluster]),
                    "Min": np.min(fills[cluster]),
                    "Max": np.max(fills[cluster]),
                }
            }
        """
        metrics = state["metrics"]
        headers = list(metrics.keys())
        row = [state["name"]]
        for v in metrics.values():
            row.append(f"{v:.2f}")
        return tb.tabulate(
            [row],
            headers=headers,
            tablefmt="fancy_grid",
            floatfmt=".2f",
            stralign="center",
            numalign="right",
        )

    def _table(self, table: Dict[str, List[Any]]) -> str:
        """Add a table with left and right headers.

        Args:
            table: Table with left header and top header, i.e.

        .. code-block:: json

            {
                "top_headers": [ "In meters", "Bin" ],
                "left_labels": [ "Position", "Target" ],
                "rows" : [[pos_m, pos], [target_m, target]]
            }
        """
        rows = table["rows"]
        if "left_labels" in table:
            left_labels = table["left_labels"]
            for i, row in enumerate(rows):
                row.insert(0, left_labels[i])
        table = tb.tabulate(
            rows,
            headers=table["top_headers"],
            tablefmt="fancy_grid",
            floatfmt=".2f",
            stralign="center",
            numalign="right",
        )
        return table

    def _print_levels(self, levels: List[Numeric]) -> None:
        levels = self._convert_to_tuples(levels)
        self.levels.set_data(levels, 120, self.lines)

    def print(self, info: Dict[str, Any]) -> None:  # type: ignore
        texts = []

        if "components" in info:
            for comp in info["components"]:
                if "rows" in comp:
                    texts.append(self._table(comp))
                elif "metrics" in comp:
                    texts.append(self._metric(comp))

        if "input" in info:
            texts.append(self._input(info["input"]))

        self.state_text.set_text("\n".join(texts))

        # Update the fill level graph if provided
        if "levels" in info:
            self._print_levels(info["levels"])


class BlankTab(Tab):
    def create(self) -> None:
        self.text = Text("This tab is currently unused.")
        self.box = create_box(self.text, "Custom Tab")
        body = Pile(
            [(HEIGHT_HEADER, self.header_box), self.box, (HEIGHT_LOGS, self.log_box)]
        )
        self._frame = AttrMap(Frame(body=body, footer=self.footer), "box gray")

    def print(self, *args, **kwargs) -> None:
        pass


TAB_REGISTER = dict(scalping=ScalpingTab, blank=BlankTab)


class TerminalInterface:
    """TUI interface for monitoring Plant<->Controller interaction.

    Args:
        port: Subscribe to ``port``, publish to ``port + 1``. This is the
              inverse of the :class:`~psipy.rl.cycle_manager.CycleManager`, as
              the two interact through these two ports in a pub/sub fashion.
    """

    # TODO: Scrolling on the status box so that long states can still be read?
    # TODO: When clearing an error occurs in the custom tab,
    #  it jumps back to the main tab
    # TODO: Cursor on buttons is ugly, potential solution:
    #  https://stackoverflow.com/questions/34633447/urwid-make-cursor-invisible

    def __init__(self, port: int, update_ms: int = 1):
        self.start_time = datetime.now()
        self.stopped = False
        self.update_s = min(1 / 1000, update_ms / 1000)

        self.last_data: Optional[Dict[str, Any]] = None
        self.log_buffer: Deque[str] = deque(maxlen=NUM_LOGS)

        # zmq setup
        LOG.info("Connect zmq...")
        port_sub, port_pub = port, port + 1
        self.zmq = zmq.Context()
        self.pub = self.zmq.socket(zmq.PUB)
        self.pub.connect(f"tcp://localhost:{port_pub}")
        self.sub = TUISubscriber(port_sub)
        self.sub.start()
        self.ports = (port_sub, port_pub)
        LOG.info(f"Connected zmq to ports pub {port_pub} and sub {port_sub}.")

        # Create tabs; custom tab is set via zmq
        self.main_tab = MainTab(self.start_time, self.ports)
        self.custom_tab: Tab = BlankTab(
            self.main_tab.header_box, self.main_tab.footer, self.main_tab.log_box
        )
        self.prev_cust_data: Dict[str, Any] = {}
        self.custom_tab_set = False

        # Received from loop on episode start.
        self.plant_name = "N.A."
        self.controller_name = "N.A."
        self.episode_number = 0
        # Set ports so that user can see ports in TUI if loop isn't running
        self.main_tab.header_text.set_text(
            f"TUI ready, awaiting loop on port {self.ports[0]}."
        )

        # Build loop
        self._build_loop()

        # Create button callbacks
        urwid.connect_signal(
            self.main_tab.main_button,
            "click",
            lambda x: self._set_main_widget(self.main_tab.frame),
        )
        urwid.connect_signal(
            self.main_tab.custom_button,
            "click",
            lambda x: self._set_main_widget(self.custom_tab.frame),
        )

    def _get_custom_info(self) -> Optional[Dict[str, Any]]:
        """Gets and updates custom information as it comes in.

        Previous information sent over the custom topic is merged with the
        given information, with the more recent information overwriting old
        keys. In this way, :meth:`step_custom` of :class:`CycleManager` can
        be called multiple times per loop. This allows sending information
        from multiple areas of the loop.
        """
        latest_data = self.sub["custom"]
        cust_info = self.prev_cust_data
        if latest_data is not None:
            cust_info.update(latest_data)
            return cust_info
        return None

    def _set_custom_tab(self, classname: Optional[str]) -> None:
        """Update the custom tab class if requested."""
        if classname is not None and not self.custom_tab_set:
            tab_class = TAB_REGISTER[classname]
            tab = tab_class(
                self.main_tab.header_box, self.main_tab.footer, self.main_tab.log_box
            )
            self.custom_tab = tab
            self.custom_tab_set = True

    def _build_loop(self) -> None:
        self.main_loop = urwid.MainLoop(
            self.main_tab.frame, palette, unhandled_input=self.handle_input
        )

    def _set_main_widget(self, widget: urwid.Widget) -> None:
        self.main_loop.widget = widget

    def start(self):
        self.main_loop.set_alarm_at(self.update_s, self.print)
        self.main_loop.run()  # Exits on keypress or error
        self.stop()

    def handle_input(self, key) -> None:
        """Capture user keyboard input and react."""
        # Exit TUI but leave everything else running
        if key == "Q" or key == "q":
            raise urwid.ExitMainLoop

        # Stop current episode, maybe start next
        if key == "S" or key == "s":
            self.pub.send_multipart([b"cmd", b"episode_stop"])

        # Exit loop
        if key == "E" or key == "e":
            self.pub.send_multipart([b"cmd", b"exit"])

        if key == "D" or key == "d":
            self.create_error_popup("User generated error.", "No traceback.")

    def _print(self) -> None:
        """Update screen."""
        # Check for exceptions first, if none, continue
        data = self.sub["exception"]
        if data is not None:
            self.create_error_popup(data["exception"], data["traceback"])

        # Check if turning the loop off was received from the CM
        data = self.sub["lifecycle"]
        if data is not None:
            event = data.pop("event")
            if event == "episode_stop":
                raise urwid.ExitMainLoop
            if event == "exit":
                raise urwid.ExitMainLoop

        # Extract data if it exists
        data = self.sub["step"]
        if data is not None:
            self.last_data = data

            # Set the header names
            self.controller_name = data["control"]
            self.plant_name = data["plant"]
            self.episode_number = data["episode"]
            # Extract relevant data (on a copy)
            data = self.last_data.copy()
            meta = data["state"].pop("meta")
            meta["Episode"] = self.episode_number
            meta["Step"] = data["step"]
            meta["Interaction Time"] = format_timer(data["interaction_time"])
            status = data["state"]
            status["action"] = data["action"]
            # Start setting text
            self.main_tab.print(
                self.plant_name, self.controller_name, status, meta, data["stats"]
            )

        # Extract custom info if it exists
        cdata = self._get_custom_info()
        if cdata is not None:
            self.last_cust_data = cdata
            cdata = self.last_cust_data.copy()
            requested_tab = cdata["tab"]
            self._set_custom_tab(requested_tab)
            self.custom_tab.print(cdata["info"])

        # Print latest log messages in tui.
        log_entry = self.sub["logs"]
        if log_entry is not None:
            self.log_buffer.append(log_entry["msg"])
            self.main_tab.log_text.set_text("".join(self.log_buffer))

    def print(self, _loop, _data) -> None:
        """Prints and resets the callback to print again after update_s."""
        self._print()
        _loop.set_alarm_in(self.update_s, self.print)

    def _acknowledge_error_callback(self, *args, **kwargs):
        """Close error window."""
        self._set_main_widget(self.main_tab.frame)

    def create_error_popup(self, error: str, traceback: str) -> None:
        """Create and display error object to overlay the populated TUI."""
        text = f"An error occurred in the interaction loop! ({error})\n\n{traceback}"
        error_text = Text(text)
        error_title = "ERROR"
        error_box = create_box(error_text, error_title, title_attr="error title")
        error_string = "Press Enter to Acknowledge"
        button = Padding(
            Filler(TextButton(error_string, self._acknowledge_error_callback))
        )
        error_items = Pile([error_box, (1, button)])
        error_frame = AttrMap(
            Overlay(
                error_items,
                self.main_tab.frame,
                align="center",
                valign="middle",
                width=("relative", 70),
                height=("relative", 70),
            ),
            "error",
        )
        self._set_main_widget(error_frame)

    def __del__(self):
        if not self.stopped:
            self.stop()

    def stop(self):
        self.zmq.destroy()
        self.sub.stop()
        self.stopped = True
        return self.stopped


def run(port, update_ms: int = 10):
    tui = TerminalInterface(port, update_ms)
    tui.start()


def main():
    parser = ArgumentParser(description="PSIORI RL Toolbox Terminal UI")
    parser.add_argument(
        "port",
        type=int,
        default=5556,
        help="I/O port to subscribe to, will publish at port+1.",
    )
    args = parser.parse_args()
    port_sub = args.port
    port_pub = port_sub + 1
    run(port_sub, port_pub)


if __name__ == "__main__":
    main()
