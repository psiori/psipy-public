# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""
.. todo::

    Module/task description.

.. todo::

    Docstrings for classes/major methods.

"""

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from psipy.rl.plant import Plant, TAction, TState

AVAILABLE_ENVIRONMENTS = (
    "CartPole-v0",
    "CartPole-v1",
    "CartPoleUnbounded-v0",
    "CartPoleUnboundedContinuous-v0",
    "CartPoleAssistedBalance-v0",
    "MountainCar-v0",
    "LunarLander-v2",
)


class GymPlant(Plant[TState, TAction], metaclass=ABCMeta):
    """Abstract base class for OpenAI's Gym based plants."""

    def __init__(
        self,
        env_name: str,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        add_action_to_state: bool = True,
        **kwargs,
    ):
        super().__init__(cost_func)
        assert env_name in AVAILABLE_ENVIRONMENTS, env_name
        try:
            self._env = gym.make(env_name, cost_func=cost_func, render_mode="human", **kwargs)
        except TypeError:
            # OpenAI original gyms do not accept kwargs
            print("ORIGINAL GYM")
            self._env = gym.make(env_name, render_mode="human")
        self._solved = False
        self._add_action = add_action_to_state

    def __del__(self):
        self._env.close()

    def _get_next_state(self, state: TState, action: TAction) -> TState:
        if isinstance(self._env.action_space, spaces.Discrete):
            action = action.as_array().astype(np.int32)[0]
        else:
            action = action.as_array()[0]
        obs, cost, terminated, truncated, info = self._env.step(action)
        if self._add_action:
            # Put the action value into the state (not the index or nodoe values)
            obs = np.append(obs, action)
        cost = self._compute_cost(state, cost)
        return self.state_type(obs, cost, terminated)

    @abstractmethod
    def _compute_cost(self, state: TState, cost: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def solve_condition(self, state: TState) -> bool:
        raise NotImplementedError

    def notify_episode_starts(self) -> bool:
        super().notify_episode_starts()
        self._solved = False
        obs, info = self._env.reset()
        if self._add_action:
            obs = np.append(obs, 0.0)
        self._current_state = self.state_type(obs, 0, False)
        return True

    def notify_episode_stops(self) -> bool:
        self._solved = self.solve_condition(self._current_state)
        return True

    def render(self) -> None:
        pass
        #self._env.render()

    @property
    def is_solved(self) -> bool:
        return self._solved
