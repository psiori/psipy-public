# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plants for control tasks within a simulated Cartpole environment.

There are three different main plants: :class:`CartPolePlant`,
:class:`CartPoleUnboundedPlant`, and :class:`CartPoleAssistedBalancePlant`.

The :class:`CartPolePlant` is the classic OpenAI cartpole balance
plant, with no edits. The task is the balance the pole and episodes terminate
if it falls.

The :class:`CartPoleUnboundedPlant` unlocks the terminal condition on pole angle and
adds friction. There are two variants, discrete and continuous. Both can run either
the swingup task or the sway control task (pass these as flags).

The :class:`CartPoleUnboundedPlant` is like the :class:`CartPolePlant`, but mimics the
hardware in Hamburg which has wooden pegs to prevent the pole from falling.
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


class CartPoleState(State):
    _channels = (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_velocity",
        "move_ACT",
    )


class CartPoleTrigState(State):
    """Cartpole state that uses trig values for the angle instead of radian/deg."""

    _channels = (
        "cart_position",
        "cart_velocity",
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        "move_ACT",
    )


class CartPoleAction(Action):
    """Parent class of cartpole actions."""
    pass


class CartPoleGymAction(CartPoleAction):
    """The action as specified in the OpenAI Cartpole gym specs"""

    channels = ("move",)
    legal_values = ((0, 1),)  # left and right


class CartPoleBangAction(CartPoleAction):
    """Action with left, right, and do nothing actions"""

    channels = ("move",)
    legal_values = ((-10, 10),)


class CartPoleGymDiscretizedAction(CartPoleAction):
    """Action with more fine grained discrete values."""

    channels = ("move",)
    legal_values = ((-20, -10, -5, 0, 5, 10, 20),)


class CartPoleContAction(CartPoleAction):
    """Continuous Cartpole action"""

    dtype = "continuous"
    channels = ("move",)
    legal_values = ((-10, 10),)  # -10,10 force


class CartPolePlant(GymPlant[CartPoleState, CartPoleGymAction]):
    """The classic cartpole task from OpenAI gym, unedited.

    Args:
        use_gym_solve: When True, the solve condition as defined by OpenAI is used.
                       This checks if the pole stayed up for the previous 200 episodes.
        cost_func: The cost function, optional
        use_renderer: When True, uses a renderer which displays the interaction time.
    """

    renderable = True
    state_type = CartPoleState
    action_type = CartPoleGymAction

    def __init__(
        self,
        use_gym_solve: bool = True,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        use_renderer: bool = False,
        **kwargs,
    ):
        env = "CartPole-v0" if not use_renderer else "CartPole-v1"
        super().__init__(env, cost_func, **kwargs)
        self.success_deque: Deque[int] = deque(maxlen=100)
        self.gym_solve = use_gym_solve

    def _compute_cost(self, state: CartPoleState, cost: float) -> float:
        cost = tanh2(state.as_array("pole_angle"), C=0.01, mu=0.05)
        # if state.terminal:
        #     cost = 1.0
        return float(cost)  # JSON serializable

    def _get_next_state(
        self, state: CartPoleState, action: CartPoleGymAction
    ) -> CartPoleState:
        print("ACTION:")
        print(action.as_dict())
        
        # SL: HACK to make sure the action is an int, not a np.float64 comming from NFQ.
        action = CartPoleGymAction({'move': int(action['move'])})
        
        state = super()._get_next_state(state, action)
        # try:
        #     self.vc.capture_frame()
        # except:  # vc was not set externally
        #     pass
        
        # DEPR: state.terminal = self._env.steps_beyond_done is not None
        # TERMINATED in gymnasium's step function makes this obsolete
        return state

    def solve_condition(self, state: CartPoleState) -> bool:
        """Achieve >= 195 average reward over 100 episodes in a row

        Reward == episode step, since originally reward was 1 per step.
        https://github.com/openai/gym/wiki/CartPole-v0
        """
        self.success_deque.append(self.episode_steps)
        if self.gym_solve:
            if np.mean(self.success_deque) >= 195:
                LOG.info(f"{self.__class__.__name__} solved!")
                return True
        else:
            if np.abs(state["pole_angle"]) < 0.1 and not state.terminal:
                return True
        return False

    def calculate_cycle_time(self, n_obs: int):
        return n_obs * self._env.tau


class CartPoleUnboundedPlant(GymPlant[CartPoleTrigState, CartPoleAction]):
    """Cartpole, but the pole is hanging down and can be freely spun."""

    renderable = True
    state_type = CartPoleTrigState
    action_type = CartPoleAction

    def __init__(
        self,
        swingup: bool = False,
        sway: bool = False,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        kwargs["swingup"] = swingup
        kwargs["sway"] = sway
        super().__init__("CartPoleUnbounded-v0", cost_func, **kwargs)
        self.swingup = swingup
        self.sway = sway
        assert not (self.swingup and self.sway), "Invalid mode."
        self.success_deque: Deque[int] = deque(maxlen=100)

    def _compute_cost(self, state: CartPoleTrigState, cost: float) -> float:
        return cost

    def _get_next_state(
        self, state: CartPoleTrigState, action: CartPoleAction
    ) -> CartPoleTrigState:
        state = super()._get_next_state(state, action)
        # try:
        #     self.vc.capture_frame()
        # except: # vc was not set externally
        #     pass
        #state.terminal = self._env.steps_beyond_done is not None
        return state

    def notify_episode_starts(self) -> bool:
        super().notify_episode_starts()
        cost = self._env.current_cost
        self._current_state.cost = cost
        return True

    def solve_condition(self, state: TState) -> bool:
        if self.swingup:
            if state["pole_cosine"] > 0.9:
                self.success_deque.append(True)
                return True
            return False
        elif self.sway:
            if state["pole_sine"] < 0.05 and abs(state["cart_position"]) < 0.1:
                self.success_deque.append(True)
                return True
            return False
        raise NotImplementedError


class CartPoleUnboundedContinuousPlant(
    CartPoleUnboundedPlant, GymPlant[CartPoleTrigState, CartPoleContAction]
):
    """Unbounded cartpole, but with continuous actions."""

    renderable = True
    state_type = CartPoleTrigState
    action_type = CartPoleContAction

    def __init__(
        self,
        swingup: bool = False,
        sway: bool = False,
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        kwargs["swingup"] = swingup
        kwargs["sway"] = sway
        GymPlant.__init__("CartPoleUnboundedContinuous-v0", cost_func, **kwargs)
        self.swingup = swingup
        self.sway = sway
        assert not (self.swingup and self.sway), "Invalid mode."
        self.success_deque: Deque[int] = deque(maxlen=100)


class CartPoleAssistedBalancePlant(GymPlant[CartPoleState, CartPoleAction]):
    """Cartpole, but with stops on the sides to prevent the pole from falling.

    This plant mimics the HH cartpole hardware with stops installed and uses just the
    direct angle value since the pole can not spin and thus does not encounter the 0
    discontinuity.
    """

    renderable = True
    state_type = CartPoleState
    action_type = CartPoleAction

    def __init__(
        self, cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None, **kwargs
    ):
        super().__init__("CartPoleAssistedBalance-v0", cost_func, **kwargs)

    def _compute_cost(self, state: CartPoleState, cost: float) -> float:
        cost = tanh2(state.as_array("pole_angle"), C=0.1, mu=0.05)
        # if state.terminal: #TODO
        #     cost = 1.0
        return float(cost)  # JSON serializable

    def _get_next_state(
        self, state: CartPoleState, action: CartPoleAction
    ) -> CartPoleState:
        state = super()._get_next_state(state, action)
        state.terminal = self._env.steps_beyond_done is not None
        return state

    def solve_condition(self, state: CartPoleState) -> bool:
        """Return True if potentially solved.

        If angle < .10 rad off vertical and pole not looping
        at end of episode, the task is potentially solved.  It may have
        just ended within the goal region, and so the saved model needs to
        be double checked.
        """
        if np.abs(state["pole_angle"]) < 0.10 and np.abs(state["pole_velocity"]) < 8:
            return True
        return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(20, 10))
    max_radians = 12 * 2 * math.pi / 360
    print(max_radians * 0.95)

    x = np.linspace(-max_radians, max_radians, 1000)
    ax.plot(x, tanh2(x, C=0.5, mu=max_radians * 0.95))
    plt.show()
