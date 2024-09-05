# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Controllers are the primary acting component inside a loop.

.. autosummary::

    Controller
    ContinuousRandomActionController
    DiscreteRandomActionController

Lifecycle::

    .__init__
    for episode in episodes:
        .notify_episode_starts
        for step in episode:
            .get_action(State) -> _get_action(np.ndarray)
        .notify_episode_stops
    .__del__

"""

import logging
import random
import sys
from abc import ABCMeta, abstractmethod
from importlib import import_module
from inspect import unwrap
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

from psipy.core.io import MemoryZipFile, Saveable
from psipy.rl.plant import Action, State
from psipy.rl.plant.plant import Numeric

__all__ = [
    "ContinuousRandomActionController",
    "Controller",
    "DiscreteRandomActionController",
]

LOG = logging.getLogger(__name__)


class Controller(Saveable, metaclass=ABCMeta):
    """Base class for controllers, all controllers implement this interface

    Args:
        state_channels: state channel names this controller sees
        action_channels: action channels this controller controls;
            defaults to all channels
        action: the plant's action type, to be produced by the controller
    """

    def __init__(
        self,
        state_channels: Tuple[str, ...],
        action: Type[Action],
        action_channels: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ):
        self.action_channels = action.channels
        if action_channels is not None:
            assert all(channel in action.channels for channel in action_channels)
            self.action_channels = action_channels
        super().__init__(
            state_channels=state_channels,
            action_channels=self.action_channels,
            **kwargs,
        )
        self.action_type = action
        self.state_channels = state_channels
        self._partial = len(self.action_channels) != len(action.channels)

    def get_action(self, state: State) -> Action:
        state_values = state.as_array(*self.state_channels)
        action_values = self._get_action(state_values).ravel()
        return self.action_type(dict(zip(self.action_channels, action_values)))

    def is_partial(self):
        return self._partial

    @abstractmethod
    def notify_episode_starts(self) -> None:
        ...

    @abstractmethod
    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop.

        Raises:
            NotNotifiedOfEpisodeStart
        """
        ...

    def _get_action(self, state: np.ndarray) -> np.ndarray:
        """Controller specific get_action."""
        ...

    def get_config(self) -> Dict:
        return self._config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _save(self, zip: MemoryZipFile) -> MemoryZipFile:
        return zip.add("config.json", self.get_config())

    @classmethod
    def _load(cls, zip: MemoryZipFile):
        return cls.from_config(zip.get("config.json"))

    @classmethod
    def load_action_type(
        cls, meta: Dict[str, str], custom_objects: Optional[List[Type[object]]] = None
    ) -> Type[Action]:
        """Get the correct action_type as defined in meta from custom_objects.

        As the specific action type a controller is used with is unknown during
        development time and only specified when a user makes use of the
        controller, it is unknown which action type to use when loading a class
        from disk. Therefore a json containing the action type's class info
        is stored with the controller and the correct action type determined
        from that information. This follows the approach Keras is taking with
        a ``custom_objects`` argument, only that we do not actually instantiate
        the custom object here, but only need the class, aka ``type(instance)``.

        If the action type is not found in the ``custom_objects`` argument, this
        method additionally checks the current ``sys.modules`` variable and
        tries to load it from there. If both fails, this raises a
        :class:`ValueError`.

        This method or an alternative to it should in the future be provided by
        :class:`~psipy.core.io.saveable.Saveable` or
        :class:`~psipy.core.io.memory_zip_file.MemoryZipFile`. For now, this
        here is the only location such a capability is required, therefore its
        implementation can be tested here.

        Args:
            meta: Content of ``meta.json`` from a :class:`Saveable` zipfile.
            custom_objects: List of custom objects to use for loading action types.

        Raises:
            ValueError: Raised when action type could not be loaded.
        """
        if custom_objects is None:
            custom_objects = list()
        custom_objects = [obj for obj in custom_objects if issubclass(obj, Action)]
        try:
            return next(
                obj
                for obj in custom_objects
                if meta["class_name"] == unwrap(obj).__name__
            )
        except StopIteration:
            pass
        try:
            import_module(meta["class_module"])
            return getattr(sys.modules[meta["class_module"]], meta["class_name"])
        except (KeyError, AttributeError):
            raise ValueError(
                f"Could not load `{meta['class_module']}.{meta['class_name']}, "
                f"as it was neither found in `{cls.__name__}.load` argument "
                f"`custom_objects`, nor in available `sys.modules`."
            )

    def preprocess_observations(self, stacks: np.ndarray) -> Any:
        """Placeholder method used within ANN-based controllers.

        See :class:`psipy.rl.control.nfq.NFQ.preprocess_observations` for more
        details.
        """
        return stacks


class MultipleController(Controller):
    """Class for multiple controllers controlling one action

    Args:
        controllers: a list of partial-action controllers
        state_channels: state channel names this controller sees
        action: the plant's action type, to be produced by the controller
    """

    def __init__(
        self,
        controllers: List[Controller],
        state_channels: Tuple[str, ...],
        action: Type[Action],
    ):
        # TODO: Does this save properly?
        super().__init__(state_channels=state_channels, action=action)
        assert all(controller.is_partial() for controller in controllers)
        self.controllers = controllers

    # TODO Str of form MultipleController(controller1, controller2, ...)
    # def __str__(self):
    #     return ""

    def get_action(self, state: State) -> Action:
        return Action.merge(*[ctrl.get_action(state) for ctrl in self.controllers])

    def notify_episode_starts(self) -> None:
        pass

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop."""
        pass

    def _get_action(self, state: np.ndarray) -> np.ndarray:
        # Just implemented to adhere to the abstract superclass.
        raise ValueError("This method should not be called. Use get_action instead.")


class ContinuousRandomActionController(Controller):
    """Controller that generates continuous random actions for the given action type

    Each sampled action is repeated for 'num_repeat' steps before a new one is sampled.
    """

    def __init__(
        self,
        state_channels: Tuple[str, ...],
        action: Type[Action],
        action_channels: Optional[Tuple[str, ...]] = None,
        num_repeat: int = 0,
    ):
        super().__init__(
            state_channels=state_channels,
            action=action,
            action_channels=action_channels,
        )
        if action.dtype != "continuous":
            raise ValueError("Attempting to continuously control discrete action type!")
        self.delay = num_repeat
        self.legal_values = self.action_type.get_legal_values(*self.action_channels)

        self._delay_count = self.delay  # force initial actions
        self._prev_action: Optional[np.ndarray] = None

    def __str__(self):
        return str(self.__class__)

    def notify_episode_starts(self) -> None:
        pass

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop."""
        self._delay_count = self.delay
        self._prev_action = None

    def _get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(len(self.legal_values))
        if self._delay_count >= self.delay or self._prev_action is None:
            for i, (low, high) in enumerate(self.legal_values):
                # Draws [low, high] assuming rounding is in your favor
                action[i] = random.uniform(low, high)
            self._delay_count = 0
            self._prev_action = action
        else:
            action = self._prev_action
        self._delay_count += 1
        return action


class DiscreteRandomActionController(Controller):
    """Controller that generates discrete random actions for the given action type

    Each sampled action is repeated for 'num_repeat' steps before a new one is sampled.
    """

    def __init__(
        self,
        state_channels: Tuple[str, ...],
        action: Type[Action],
        action_channels: Optional[Tuple[str, ...]] = None,
        num_repeat: int = 0,
    ):
        super().__init__(
            state_channels=state_channels,
            action=action,
            action_channels=action_channels,
        )
        if action.dtype != "discrete":
            raise ValueError("Attempting to discretely control continuous action type!")
        self.delay = num_repeat
        self.legal_values = self.action_type.get_legal_values(*self.action_channels)

        self._delay_count = self.delay  # force initial actions
        self._prev_action: Optional[Dict[str, Numeric]] = None

    def __str__(self):
        return str(self.__class__)

    def notify_episode_starts(self) -> None:
        pass

    def notify_episode_stops(self) -> None:
        """Handles post-episode cleanup, called from the loop."""
        self._delay_count = self.delay
        self._prev_action = None

    def get_action(self, state: State) -> Action:
        action_dict = {}
        additional = {}

        if self._delay_count >= self.delay or self._prev_action is None:
            for action, values in zip(self.action_channels, self.legal_values):
                try:
                    index = random.sample(range(len(values)), 1)[0]
                    additional[f"{action}_index"] = index
                    action_dict[action] = values[index]
                except ValueError as e:
                    LOG.error("Is your action space a tuple of sequences? := ((x,y),)")
                    raise e
            self._delay_count = 0
            self._prev_action = action_dict
        else:
            action_dict = self._prev_action
        self._delay_count += 1

        return self.action_type(action_dict, additional_data=additional)


if __name__ == "__main__":
    from psipy.rl.plant.tests.mocks import MockAction, MockState

    cc = ContinuousRandomActionController(
        MockState.channels(), MockAction, action_channels=(MockAction.channels[0],)
    )
    dc = DiscreteRandomActionController(
        MockState.channels(), MockAction, action_channels=(MockAction.channels[1],)
    )
    mc = MultipleController([cc, dc], MockState.channels(), MockAction)
    state = MockState(np.array([0] * len(MockState.channels())))
    for _ in range(10):
        a = mc.get_action(state)
        print(a)
