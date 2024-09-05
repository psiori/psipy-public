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
import math
from collections import deque
from typing import Callable, Deque, Optional

import numpy as np

from psipy.rl.control.nfq import tanh2
from psipy.rl.plant import Action, State, TState
from psipy.rl.plant.gym.gym_plant import GymPlant

LOG = logging.getLogger(__name__)


class LunarLanderState(State):
    _channels = (
        "x_coordinate",
        "y_coordinate",
        "x_velocity",
        "y_velocity",
        "angle",
        "angular_velocity",
        "left_leg_contact",
        "right_leg_contact",
        "fire_engine_ACT"
    )


class LundarLanderDiscreteAction(Action):
    """The action as specified in the OpenAI LunarLander gym specs"""

    channels = ("fire_engine",)
    # Noop, Left, Main, Right
    legal_values = ((0, 1, 2, 3),)


class LunarLanderPlant(GymPlant[LunarLanderState, LundarLanderDiscreteAction]):
    """The Lunar Lander task from OpenAI gym, unedited.

    Args:
        use_gym_solve: when True, the solve condition as defined by OpenAI is used.
        cost_func: the cost function, optional
    """

    renderable = True
    state_type = LunarLanderState
    action_type = LundarLanderDiscreteAction

    def __init__(
        self,
        use_gym_solve: bool = True,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs
    ):
        super().__init__("LunarLander-v2", cost_func, **kwargs)
        self.success_deque: Deque[int] = deque(maxlen=100)
        self.gym_solve = use_gym_solve

    def _compute_cost(self, state: LunarLanderState, cost: float) -> float:
        # print("COST:", -cost/1000)
        return -cost/1000  # cost comes in as a complex reward; turn it into a cost

    def _get_next_state(
        self, state: State, action: LundarLanderDiscreteAction
    ) -> LunarLanderState:
        state = super()._get_next_state(state, action)
        return state

    def solve_condition(self, state: LunarLanderState) -> bool:
        """Achieve >= 200 average reward over 100 episodes in a row

        Reward for moving from the top of the screen to the landing pad and zero speed
        is about 100..140 points. If the lander moves away from the landing pad it
        loses reward. The episode finishes if the lander crashes or comes to rest,
        receiving an additional -100 or +100 points. Each leg with ground contact is
        +10 points. Firing the main engine is -0.3 points each frame. Firing the side
        engine is -0.03 points each frame.
        https://gym.openai.com/envs/LunarLander-v2/
        """
        if self.gym_solve:
            self.success_deque.append(self.episode_steps)
            if np.mean(self.success_deque) >= 200:
                LOG.info(f"{self.__class__.__name__} solved!")
                return True
        else:
            # Velocities are 0, and is upright, and position is inside the flags
            correct_x = -.2 < state["x_coordinate"] < .2
            correct_y = -.2 < state["y_coordinate"] < .2
            correct_velocity = (
                state["x_velocity"]
                == state["y_velocity"]
                == state["angular_velocity"]
                == 0
            )
            correct_angle = state["angle"] == 0
            correct_leg_contact = (
                state["left_leg_contact"] == state["right_leg_contact"] == 1
            )
            if (
                correct_x
                and correct_y
                and correct_velocity
                and correct_angle
                and correct_leg_contact
            ):
                return True
        return False

    def calculate_cycle_time(self, n_obs: int):
        return n_obs * 50  # sim runs at 50 FPS
