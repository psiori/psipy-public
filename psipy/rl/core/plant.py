"""Plants can be anything from factories over simulations to computer games.

.. todo::

    More extensive explanation of some of the concept behind Plant and its
    lifecycle methods.

.. todo::

    Intro to Action, State and (optional) TerminalStates.

.. todo::

    Runnable examples / doctests for all components!

.. autosummary::

    Action
    State
    TerminalStates
    Plant

"""

import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from math import isfinite, isnan
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np

from psipy.core.utils import flatten
from psipy.rl.core.cycle_manager import CM

__all__ = ["Action", "Plant", "State", "TerminalStates"]


LOG = logging.getLogger(__name__)
Numeric = Union[int, float]


TState = TypeVar("TState", bound="State")
TAction = TypeVar("TAction", bound="Action")


class Action(metaclass=ABCMeta):
    """Abstract base class for Action objects.

    Action objects hold all information about how a controller manipulates a
    plant. An action can be made up of multiple channels, each for example
    representing a single control value to be passed to a simulator.

    .. todo::

        Add examples for both normal as well as development usage (using None).
        Relevant discussion:
        https://github.com/psiori/psipy-rl/pull/74/files/fbc58c7#diff-e5b1272aecfc92f0854aa4292ac9e9fd

    Args:
        data: Data of the action, per channel as array or dict.
        additional_data: Additional "`action channels`", commonly metadata
                         of the controller again required for training.

    """

    __slots__ = ["_data", "_additional_data"]

    dtype: ClassVar[str] = "discrete"
    channels: ClassVar[Tuple]
    semantic_channels: ClassVar[Optional[Tuple[str, ...]]] = None

    #: Legal values per action
    #: Defined by lower and upper bounds for continuous action spaces,
    #: and actual action options for discrete spaces
    legal_values: ClassVar[Tuple[Sequence, ...]]

    #: Default values for action channels
    #: One default-value per channel needs to be defined
    default_values: ClassVar[Tuple[float, ...]]

    #: The actual action data, one value per channel.
    _data: Dict[str, Numeric]

    #: Additional "`action channels`", commonly metadata of the controller
    #: which it might require again for training.
    _additional_data: Dict[str, Numeric]

    def __init__(
        self,
        data: Union[np.ndarray, Sequence[Numeric], Mapping[str, Numeric]],
        additional_data: Optional[Mapping[str, Numeric]] = None,
    ) -> None:
        # Actions passed as explicit key-value pairs. May be partial.
        if isinstance(data, dict):
            if set(data.keys()).difference(self.channels):
                raise ValueError(f"Received unknown action channels in {data}")
            channels = [c for c in self.channels if c in data]
            data = dict((c, data[c]) for c in channels)

        # Actions passed as individual values. Has to be one value per channel.
        else:  # elif isinstance(values, (np.ndarray, list))
            if len(data) != len(self.channels):
                raise ValueError("Expected a single value for each action channel.")
            data = dict(zip(self.channels, np.asarray(data).ravel()))

        self._data = {
            channel: self._coerce_dtype(value)
            for channel, value in data.items()
            if value is not None
        }
        self._partial = any(channel not in self._data for channel in self.channels)

        self._additional_data = dict()
        if additional_data is not None:
            self._additional_data = dict(additional_data)
        assert set(self.channels).isdisjoint(self._additional_data.keys())

        # Check for nans first, then check for illegal values
        if not self.is_partial():
            if any(np.isnan(np.asarray(v).sum()) for v in self._data.values()):
                raise ValueError(f"Action contains NaNs: {data}.")
        self._check_illegal_input()

        # Double check sub-class definition.
        assert self.dtype in ["discrete", "continuous"], "Invalid action space type!"
        if self.dtype == "continuous":
            # Expecting [lower, upper] bounds for continuous actions
            assert all(len(legal_pair) == 2 for legal_pair in self.legal_values)
        assert isinstance(self.legal_values, tuple)
        if self.semantic_channels is not None:
            assert len(self.semantic_channels) == len(self.channels)

    @staticmethod
    def _coerce_dtype(value: Union[Numeric, np.generic]) -> float:
        """Make sure datatype is ja python basic datatype and jsonfy-able."""
        try:
            return value.item()  # type: ignore
        except AttributeError:
            return value

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def is_partial(self) -> bool:
        return self._partial

    def items(self):
        """Iterates over the action's (channel, value) pairs in dict-like fashion."""
        return self._data.items()

    def __getitem__(self, key: str):
        """Gets a single channel's value from the action."""
        return self._data[key]

    def __len__(self):
        """Returns the total number of channels in this action, filled or empty."""
        return len(self.channels)

    def __repr__(self):
        return f"{self.name}({self._data})"

    def __str__(self):
        if self.semantic_channels is not None:
            return f"{self.name}({dict(self.as_semantic_dict())})"
        return self.__repr__()

    def __eq__(self, other):
        if other is self:
            return True
        if isinstance(other, self.__class__):
            return np.array_equal(self.as_array(), other.as_array())
        return False

    def keys(self, with_additional: bool = False) -> Tuple[str, ...]:
        """Returns the sequence of actually filled channels."""
        keys = self._data.keys()
        filled = tuple(channel for channel in self.channels if channel in keys)
        if with_additional:
            filled += tuple(self._additional_data.keys())
        return filled

    @classmethod
    def get_legal_values(cls, *channels: str) -> Tuple[Sequence, ...]:
        """Gets the legal values for specific channels."""
        legal_values = dict(zip(cls.channels, cls.legal_values))
        return tuple([legal_values[channel] for channel in channels])

    @classmethod
    def get_default_values(cls, *channels: str) -> Tuple[float, ...]:
        default_values = dict(zip(cls.channels, cls.default_values))
        return tuple([default_values[channel] for channel in channels])

    def _check_illegal_input(self):
        """Check whether input values are outside their legal ranges"""
        legal_by_channel = dict(zip(self.channels, self.legal_values))
        if self.dtype == "discrete":
            for channel, value in self._data.items():
                if value not in legal_by_channel[channel]:
                    raise ValueError(
                        f"Channel {channel} is outside legal values! "
                        f"{value} not in {legal_by_channel[channel]}"
                    )
        else:  # is continuous
            for channel, value in self._data.items():
                low, high = legal_by_channel[channel]
                if not (low <= value <= high):
                    raise ValueError(
                        f"Channel {channel} is outside legal values! "
                        f"{value} not in range {legal_by_channel[channel]}"
                    )

    def as_array(self, *channels: Union[Sequence[str], str, None]) -> np.ndarray:
        """Returns requested channel values as array.

        Only valid for actions in which all channels are equal dimensions,
        otherwise they cannot be coerced into a single array.

        Args:
            channels: Optional sequence of channels to extract into array.
        """
        if len(channels) == 0 or channels[0] is None:
            if self.is_partial():
                raise ValueError("Action is partial, cannot convert to dense array.")
            channels = self.channels
        channels = flatten(channels)
        return np.asarray([self[channel] for channel in channels])

    def as_dict(self, with_additional: bool = False):
        """Returns the action's values as dictionary.

        Args:
            with_additional: Treat the :attr:`_additional_data` as normal action
                             channels, merging them into the same returned dict.
        """
        data = self._data.copy()
        if with_additional:
            data.update(self._additional_data)
        return data

    def as_semantic_dict(self):
        if self.semantic_channels is not None:
            return OrderedDict(
                (self.semantic_channels[self.channels.index(channel)], value)
                for i, (channel, value) in enumerate(self._data.items())
            )
        return self.as_dict()

    @staticmethod
    def merge(*actions: "Action") -> "Action":
        """Merges multiple partial actions into a single action of the same type.

        Checks to assure mergability:

        - All actions are of the same class
        - All actions are partial
        - No actions overlap
        - There are no unfilled action values

        Args:
            actions: any number of partially filled actions.
        """
        if len(actions) == 1:
            if actions[0].is_partial():
                raise ValueError("Received a single incomplete action.")
            return actions[0]

        action_type = type(actions[0])
        if any(type(action) is not action_type for action in actions):
            raise ValueError("Not all actions are of the same type.")
        if any(not action.is_partial() for action in actions):
            raise ValueError("Not all actions are partial.")
        keys = sorted(flatten([action.keys() for action in actions]))
        if sorted(list(set(keys))) != keys:
            raise ValueError("Passed partial actions have overlapping channels.")

        # Merge action and make sure it is not partial.
        data = actions[0].as_dict()
        for action in actions[1:]:
            data.update(action.as_dict())

        # Merge additional data without any additional checks.
        additional_data = actions[0]._additional_data
        for action in actions[1:]:
            additional_data.update(action._additional_data)

        action = action_type(data, additional_data=additional_data)
        if action.is_partial():
            raise ValueError("Merged action is still partial.")
        return action


class State(metaclass=ABCMeta):
    """Observation type specific to a :class:`Plant`.

    Currently only supports vectors. Future plans involve extending
    :class:`State` to allow for non-vector values such as images or a mix of
    images and measurements.

    Args:
        values: Values for the State instance. Preferably to be passed as a
                dictionary mapping channels (which match State channels) to
                values.
        cost: Cost the specific observation incurred while it was generated.
        terminal: Whether this observation ended an interaction trajectory.
        meta: Additional metadata to be stored on the state object. Any
              meta information stored here should be directly related to that
              specific observation.
        check_unexpectedness: Whether to check for channels which were not
                              expected. Unexpected channels are those which
                              are passed by not listed in the State's channel
                              list.
        filler: A dictionary of State channel value mappings to employ as
                default values for any non parseable data passed in ``values``.
    """

    __slots__ = ["_data", "cost", "terminal", "meta"]

    #: Lists all channels contained in the State. Either as a tuple of strings
    #: or as a dictionary with the channel tags as keys and additional
    #: metadata in a dict as values.
    _channels: ClassVar[Union[Tuple[str, ...], Dict[str, Dict[str, Any]]]]

    #: List of channels to be ignored. Used for handling unexpected channels.
    _ignore_channels: ClassVar[Tuple[str, ...]] = tuple()

    #: The State's data, representing measurements at a single timestep from the plant.
    #: This object may be unordered and therefore should never be used directly!
    _data: Dict[str, float]

    #: Immediate, at runtime plant-prescribed cost at the current timestep.
    cost: float

    #: Whether the current observation, at runtime, ended a trajectory actively.
    #: An observation may be the last of an episode without having ended the
    #: trajectory when for example ``max_steps`` was reached.
    terminal: bool

    #: Additional meta information provided by the plant.
    meta: Dict[str, Optional[Union[Numeric, str]]]

    def __init__(
        self,
        values: Union[Dict[str, Union[np.ndarray, Numeric]], np.ndarray],
        cost: Union[int, float] = 0.0,
        terminal: bool = False,
        meta: Optional[Dict[str, Optional[Union[Numeric, str]]]] = None,
        check_unexpectedness: bool = False,
        filler: Optional[Dict[str, Numeric]] = None,
    ) -> None:
        if filler is None:
            filler = dict()
        if isinstance(values, (np.ndarray, list)):
            # Receive individual values for each channel in the form of a single
            # ndarray vector. This only works with State implementations which
            # do not make use of :meth:`enrich`.
            assert len(values) == len(self), f"{len(values)}=={len(self)}"
            data = {
                k: self._coerce_dtype(val, default=filler.get(k), name=k)
                for k, val in zip(self.keys(), values)
            }
        elif isinstance(values, dict):
            # Receive the values for channels in the form of a dictionary,
            # similar to the dictionary stored in :attr:`_data`. Important to
            # note that the passed dictionary does not need to contain all
            # channels listed in :attr:`channels`, as some remaining channels
            # might be filled by :meth:`enrich`.
            incoming_channels = tuple(values.keys())
            data = {
                k: self._coerce_dtype(values[k], name=k, default=filler.get(k))
                for k in self.keys()
                if k in incoming_channels
            }
            if check_unexpectedness:
                unexpected = tuple(set(incoming_channels) - set(data.keys()))
                # channels to be ignored are expected
                unexpected = tuple(set(unexpected) - set(self.ignore_channels()))
                if unexpected:
                    LOG.info(f"Received unexpected channels: {unexpected}")
        else:
            raise ValueError(f"Unknown values datatype, received {type(values)}.")

        self._data = self.enrich(data)
        assert set(self._data.keys()).issuperset(self.required_channels()), (
            "Some required channels are missing. Channel difference: "
            f"{set(self._data.keys()).symmetric_difference(self.keys())}"
        )
        assert all(not isnan(v) for v in self.values()), f"NaN in state: {self._data}"

        self.meta = meta or dict()
        self.cost = float(cost)
        self.terminal = bool(terminal)

    @staticmethod
    def _coerce_dtype(
        value: Any, *, default: Union[None, Numeric] = 0.0, name: str = "unnamed"
    ) -> float:
        """Converts ``value`` to :class:`float`.

        For non-coerceable types the ``default`` value is returned.

        Args:
            value: Value to coerce to :class:`float`.
            default: Value to return if ``value`` is not coerceable.
                     Purposefully set to ``0.0`` by default in order to never
                     raise errors here. ``None`` value also results in ``0.0``.
                     Method will warn when falling back to default.
            name: Name of the variable, only used for meaningful warnings.
        """
        if default is None:
            default = 0.0

        # Try casting to float.
        got_err = False
        val = value
        try:
            val = float(value)
        except (ValueError, TypeError):
            got_err = True

        # Handle non-castable values and `nan` and `inf` by falling back to default.
        if got_err or not isfinite(val):
            LOG.warning(f"Using {default} as default for {name}, got {value}.")
            val = State._coerce_dtype(default, name=name)

        return val

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __len__(self) -> int:
        return len(self.keys())

    def __repr__(self) -> str:
        return f"{self.name}({self._data})"

    def __str__(self):
        return str(self.as_dict(semantic=True))

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        return False

    @classmethod
    def channels(cls) -> Tuple[str, ...]:
        if isinstance(cls._channels, dict):
            return tuple(cls._channels.keys())
        return tuple(cls._channels)

    @classmethod
    def required_channels(cls) -> Tuple[str, ...]:
        if isinstance(cls._channels, dict):
            return tuple(
                key
                for key, meta in cls._channels.items()
                if not meta.get("optional", False)
            )
        return tuple(cls._channels)

    @classmethod
    def optional_channels(cls) -> Tuple[str, ...]:
        if isinstance(cls._channels, dict):
            return tuple(
                key
                for key, meta in cls._channels.items()
                if meta.get("optional", False)
            )
        return tuple()

    @classmethod
    def ignore_channels(cls) -> Tuple[str, ...]:
        """Returns channels to be ignored"""
        return cls._ignore_channels

    @classmethod
    def semantic_channels(cls) -> Tuple[str, ...]:
        """Lists channel descs, falling back to channel names."""
        if isinstance(cls._channels, tuple):
            return cls.channels()
        semantics = tuple(
            cls._channels[channel].get("desc", channel) for channel in cls.channels()
        )
        return semantics

    def keys(self) -> Tuple[str, ...]:
        """Provides a list of channels.

        By implementing both :meth:`keys` and :meth:`__getitem__` instances of
        this class can be used with the :class:`dict` constructor.

            >>> class MyState(State):
            ...   _channels = ("a", "b")
            >>> state = MyState(np.array([1, 2]))
            >>> state["a"]
            1.0
            >>> dict(state)
            {'a': 1.0, 'b': 2.0}
            >>> dict(**state)
            {'a': 1.0, 'b': 2.0}

        """
        return self.channels()

    def values(self) -> Tuple[float, ...]:
        """Returns all values contained in the observation in channel-order."""
        return tuple(self._data[channel] for channel in self.keys())

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def __contains__(self, channel: str) -> bool:
        """Checks whether the state contains a specific channel."""
        return channel in self.keys()

    def __getitem__(self, channel: str) -> Numeric:
        """Returns a single channel's value from the values ndarray."""
        return self._data[channel]

    def __setitem__(self, channel: str, value: Numeric) -> None:
        """Updates a single channel's value, also re-triggering value enrichment.

        .. note::

            Mutating :class:`State` instances is bad practice and strongly
            discouraged because it makes code harder to reason about. It is
            considered to fully inhibit :class:`State` instance modifications.

        """
        self._data[channel] = value
        self._data = self.enrich(self._data)

    def as_dict(self, semantic: bool = False):
        """Returns state key/values, cost, terminal as dictionary

        Args:
            semantic: Uses channel descriptions as the keys
        """
        channel_keys = self.semantic_channels() if semantic else self.channels()
        return OrderedDict(
            values=OrderedDict(
                (channel_keys[i], self[channel])
                for i, channel in enumerate(self.channels())
            ),
            cost=self.cost,
            terminal=self.terminal,
            meta=self.meta,
        )

    def as_array(self, *channels: Union[Sequence[str], str, None]) -> np.ndarray:
        """Returns requested channel values as array.

        Args:
            channels: Optional sequence of channels to extract into array.
        """
        if len(channels) == 0 or channels[0] is None:
            return np.asarray(self.values())
        channels = flatten(channels)
        return np.asarray([self[channel] for channel in channels])

    def annotate(self: TState, cost: Numeric, terminal: bool) -> TState:
        """Adds cost and terminal information to the State."""
        self.cost = float(cost)
        self.terminal = bool(terminal)
        return self

    def copy(self: TState) -> TState:
        """Return a deep copy of the original state."""
        return self.__class__(self._data, self.cost, self.terminal, self.meta)

    @classmethod
    def enrich(cls, data: Dict[str, Numeric]) -> Dict[str, Numeric]:
        """Enriches the instance's ``data`` by additional channels.

        .. note::

            This method may mutate the incoming dict instead of creating a deep
            copy to return. Additionally, while the incoming ``data`` dict might
            (or might not) be in the same order as :attr:`channels`, that
            ordered does not have to be retained during enrichment and may
            therefore be violated by the returned dict.

        Args:
            data: A :class:`State` instance's internal data.
        """
        return data

    @classmethod
    def get_semantic_channels(cls, *tags: str) -> Tuple[str, ...]:
        mapping = dict(zip(cls.channels(), cls.semantic_channels()))
        return tuple(mapping[tag] for tag in tags)


class TerminalStates(metaclass=ABCMeta):
    """Terminal state checks for a specific :class:`Plant`.

    Inherit from this class and add methods of the following signature::

        def terminal_condition_name(state):
            return True/False

    The method names should describe the terminal condition because they will
    be logged if they are met.

    To use, call the determine_if_terminal method directly from the class and
    pass in the state.  All implemented terminal methods will be checked, and
    if the condition is met, logged to stdout.  If any are met, True is
    returned, else False.

    Internal state can be maintained by declaring class level variables
    (e.g. counters) in the child class.

    Args:
        disabled: True if terminal states should not be checked for
    """

    disabled: bool

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """Initializes the subclasses upon first creation.

        Since this is creating a class level variable, initializing the subclass
        once is enough to prepopulate the methods needed. For more details, see
        https://docs.python.org/3/reference/datamodel.html#customizing-class-creation
        """
        super().__init_subclass__(**kwargs)
        cls.disabled = False
        cls.method_list = [
            getattr(cls, name)
            for name in dir(cls)
            if not str(name).startswith("__")
            and not str(name) == "determine_if_terminal"
            and not str(name) == "disable"
            and callable(getattr(cls, name))
        ]

    @classmethod
    def determine_if_terminal(cls, state: State):
        """Loops through all methods implemented in the class and evaluates them

        Methods should only take state as a parameter and return True/False based
        on terminality.
        """
        if cls.disabled:
            return False
        is_terminal = False
        for method in cls.method_list:  # type: ignore
            terminal = method(cls, state)
            if terminal:
                is_terminal = True
                LOG.info(f"Terminal state via '{method.__name__}'.")
        return is_terminal

    @classmethod
    def disable(cls):
        """Disables terminal states."""
        # TODO: This is dangerous because it alters the Class and not the Instance!
        LOG.info("Terminal states disabled.")
        cls.disabled = True


class Plant(Generic[TState, TAction], metaclass=ABCMeta):
    """The primary interface between the real world and :mod:`psipy.rl`.

    Lifecycle::

        .__init__
        for episode in episodes:
            .notify_episode_starts
            .check_initial_state
            for step in episode:
                .get_next_state
            .notify_episode_stops
        .__del__

    Args:
        cost_function: Cost function that maps states to costs. If provided,
                       will override the cost coming from the plant's implementation in the returned current state. Please be
                       aware this cost function DIFFERS from the one that is used in controllers. This cost function uses A) the plant
                       internal state representation and B) is not vectorized,
                       but expects a single state of TState as input. If you'd
                       like to use the same cost function for both controllers and plants, consider using the convenience function
                       :meth:`cost_func_wrapper` to wrap your cost function which you can find further below in this file.

                       Note: the standard statistics implemented in the Loop
                       use the costs comming from the plant's implementation,
                       not the controller's cost function. Thus, it could be
                       worthwile passing the same cost function to your plant
                       in case you do not use the 'native' costs of the plant
                       but use a custom one with your controllers.
    """

    renderable: ClassVar[bool] = False
    state_type: ClassVar[Type[TState]]
    action_type: ClassVar[Type[TAction]]
    meta_keys: ClassVar[Tuple[str, ...]] = ()

    _current_state: TState
    _episode_steps: int

    def __init__(self, cost_function: Optional[Callable[[TState], float]] = None):
        self._cost_function = cost_function

    def cycle_started(self):
        if "plant" in CM.pubsub.cmds:
            self.user_input(**CM.pubsub.cmds.pop("plant"))

    def user_input(self, **kwargs):
        if kwargs:
            LOG.warning(f"Unhandled user input kwargs: {kwargs}")

    def get_next_state(self, state: TState, action: TAction) -> TState:
        self._current_state = self._get_next_state(state, action)
        # Check if a cost function first exists (init may not have been called) and
        # if so, if it is not None in order to apply the cost function
        if hasattr(self, "_cost_function") and self._cost_function is not None:
            self._current_state.cost = self._cost_function(self._current_state)
        self._episode_steps += 1
        return self._current_state

    @abstractmethod
    def _get_next_state(self, state: TState, action: TAction) -> TState:
        """Plant specific transform of (state, action) -> state

        If you desire to enrich the state with any values, do so here as a dict update
        before casting to the StateType.  For example::

            obs = dict(something)
            obs.update({f"{k}_ACT": v for k, v in action_dict.items()})

        """
        raise NotImplementedError

    def check_initial_state(self, state: TState) -> TState:
        assert self.episode_steps == 0
        return self._current_state

    def notify_episode_starts(self) -> bool:
        self._episode_steps = 0
        self._in_terminal = False
        return True

    @abstractmethod
    def notify_episode_stops(self) -> bool:
        raise NotImplementedError

    def validate_next_state(
        self, state: TState, action: TAction, next_state: TState
    ) -> bool:
        return True

    def is_terminal(self, state: TState) -> bool:
        assert state == self._current_state
        return self._current_state.terminal

    def get_cost(self, state: TState) -> float:
        assert state == self._current_state
        return self._current_state.cost

    def render(self) -> None:
        if not self.renderable:
            raise ValueError("Plant `%s` is not renderable." % str(self))
        raise NotImplementedError

    @property
    def channels(self) -> Union[None, Tuple[str, ...]]:
        return self.state_type.channels()

    @property
    def actions(self) -> Type[Action]:
        return self.action_type

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @classmethod
    def cost_func_wrapper(
        cls, cost_func: Callable[[np.ndarray], np.ndarray], state_channels: List[str]
    ) -> Callable[[TState], float]:
        """convenience function that wraps a vecotrized cost function that is used with controllers to accept a single state object instead, thus, making it compatible with plant-internal state representations and the plant's expectations on a cost function. If you have a cost function that is working with your
        controllers, you can use this wrapper to make it compatible and pass it
        to your plants constructor.

        Args:
            cost_func: A vectorized cost function that takes a numpy array of several states and returns a numpy array of costs.
            state_channels: A list of channel names that the cost function expects in the given order. Usually the same list of channels that
            you pass on to controller.fit() or Batch.from_hdf5().

        Example:
            plant = MyPlant(cost_function=Plant.cost_func_wrapper(my_cost_func, ["position", "velocity"]))
        """

        def wrapped_cost_func(state: TState) -> float:
            state_array = state.as_array(state_channels)
            return cost_func(np.asarray([state_array]))[0]

        return wrapped_cost_func


if __name__ == "__main__":
    from psipy.rl.plants.tests.mocks import MockDiscreteAction

    a = MockDiscreteAction({"channel1": 1})

    class S(State):
        _channels = ("one", "two")

    s = S(np.array([1, None]))
    print(s)
