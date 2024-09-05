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

import logging
from collections import deque
from typing import Callable, Deque, Optional

import numpy as np

from psipy.rl.control.nfq import tanh2
from psipy.rl.plant import Action, State
from psipy.rl.plant.gym.gym_plant import GymPlant

LOG = logging.getLogger(__name__)


class MountainCarState(State):
    _channels = (
        "position",
        "velocity",
        "push_ACT",
    )


class MountainCarAction(Action):
    """The action as specified in the OpenAI Mountaincar gym specs"""

    channels = ("push",)
    legal_values = ((0, 1, 2),)  # left, nothing, right


class MountainCarPlant(GymPlant[MountainCarState, MountainCarAction]):
    renderable = True
    state_type = MountainCarState
    action_type = MountainCarAction

    def __init__(self, cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        super().__init__("MountainCar-v0", cost_func)
        self.success_deque: Deque[int] = deque(maxlen=100)

    def _compute_cost(self, state: MountainCarState, cost: float) -> float:
        cost = tanh2(state.as_array("position") - 0.5, 0.1, 1)
        if state.terminal:
            cost = 1.0
        return float(cost)  # JSON serializable

    def _get_next_state(
        self, state: MountainCarState, action: MountainCarAction
    ) -> MountainCarState:
        state = super()._get_next_state(state, action)
        return state

    def solve_condition(self, state: MountainCarState) -> bool:
        """Reach the top of the right hill (position>=.5)."""
        if state["position"] >= 0.5:
            return True
        return False
