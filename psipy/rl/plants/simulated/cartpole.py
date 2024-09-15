import math
import random
from typing import Callable, Optional, Tuple, Union
import numpy as np
from psipy.rl.core.plant import Action, Plant, State
from psipy.rl.controllers.nfq import tanh2


FLOATMAX = np.finfo(np.float32).max
PRECISION = 6
USE_FRICTION = True

def polespeed(second, first):
    return np.tan(second-first)

class CartPoleAction(Action):
    """Parent class of cartpole actions."""
    pass

class CartPoleBangAction(CartPoleAction):
    """Action with left actions right actions"""

    channels = ("move",)
    legal_values = ((-10, 10),)

class CartPoleState(State):
    """Cartpole state that uses trig values for the angle instead of radian/deg."""

    _channels = (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        "move_ACT",
    )

def default_cost_function(state: np.ndarray) -> np.ndarray:
    x, x_dot, theta, sintheta, costheta, theta_dot, move_ACT = state

    if (abs(x) > 2.4):
        return 1

    return tanh2(theta, C=0.1, mu=0.05)

class CartPole(Plant[CartPoleState, CartPoleAction]):
    """Cartpole with no angle terminals, allowing the pole to move freely.

    Environment with a pole on a cart. Can be used for both a sway goal or
    a swing up goal, or something else. Behavior depends on the cost function
    used.

        * Swing Up: cart should be moved to a specific static goal position
            and balance the pole vertically upwards.

    Note that the internal state includes the angle, theta, but that is NOT returned
    in the state.

    Observation::

        Type: Box(5)
        Num	Observation               Min         Max
        0   Cart Position             -2.4        2.4
        1   Cart Velocity             -Inf        Inf
        2   Sine of Pole Angle        -1          1
        3   Cosine of Pole Angle      -1          1
        4   Pole Velocity At Tip      -Inf        Inf

    Actions (2 types)::

        Type: Discrete
        Force in [x, y, z, ...], dependent on defined values in CartPoleUnboundedAction

        Type: Continuous
        Force in [-x,x], dependent on defined values in CartPoleUnboundedContAction

    Cost: Cost is based on a function of state.

    Starting State: Random cart location, some cart velocity, some random pole
    angle offset from facing downwards, some random pole velocity.

    Episode Termination: Cart leaves boundaries of screen (+/- 2.4).
    """

    continuous: bool = False
    steps_beyond_done: Optional[int]
    state: Tuple[float, float, float, float, float, float]


    def __init__(
        self,
        cost_function: Optional[Callable[[np.ndarray], np.ndarray]] = default_cost_function,
    ):
        super().__init__(cost_function)

        # Sim constants
        self.gravity = 9.81
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        # Friction coefficients
        self.frictioncart = 0.01 * USE_FRICTION  # 5e-4 * USE_FRICTION
        self.frictionpole = 0.01 * USE_FRICTION  # 2e-6 * USE_FRICTION

        # Goal state definition
        self.x_dot_goal = 0
        self.theta_goal = math.pi
        self.theta_dot_goal = 0

        # Legal observation and action observation
        self.max_cart_speed = 1.0
        low = np.array([-4.8, -self.max_cart_speed, -np.inf, -np.inf, -1, 1])
        high = np.array([4.8, self.max_cart_speed, np.inf, np.inf, -1, 1])
    #   self.observation_space = spaces.Box(low, high, dtype=np.float32)
    #    if self.continuous:
    #        self.action_space = spaces.Box(
    #            np.array([-np.inf]), np.array([np.inf]), dtype=np.float32
    #        )
    #    else:
    #        # 999 is a placeholder to avoid having to pass in a value
    #        self.action_space = spaces.Discrete(999)


        self.reset()

    def _get_next_state(self, state: CartPoleState, action: CartPoleAction) -> CartPoleState:
        """Apply action and step the environment tau seconds into the future.

        The majority of this code is replicated from the OpenAI gym cartpole code.
        The following additions were made:

            * Added friction
            * Removed angle terminals
            * Allow for continuous as well as discrete actions
            * Change the angle definition from degrees (which has a 0 discontinuity) to
              sine and cosine values.
            * Forces the use of semi-euler physics calculation, which seems more realistic.

        """
        #TODO: assert state == current_state 
        info = {}
        x, x_dot, theta, sintheta, costheta, theta_dot, move_ACT = self._current_state.as_array()

        force = action["move"]
        self.current_force = force

        # costheta = math.cos(theta)
        # sintheta = math.sin(theta)
        temp_top = (
            force
            + self.polemass_length * theta_dot * theta_dot * sintheta
            - self.frictioncart * np.sign(x_dot)
        )
        temp_bottom = self.total_mass
        temp = temp_top / temp_bottom
        thetaacc_top = (
            self.gravity * sintheta
            - costheta * temp
            - self.frictionpole * theta_dot / self.polemass_length
        )
        thetaacc_bottom = self.length * (
            4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass
        )
        thetaacc = thetaacc_top / thetaacc_bottom
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Must use semi-implicit euler here! Otherwise the pole's accel will diverge!
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot
        sin = math.sin(theta)
        cos = math.cos(theta)

        # Penalize strongly going off screen or high pole
        terminal = False
        if abs(x) > 2.4:
            terminal = True

        info["force"] = force
        info["theta_accel"] = thetaacc
        info["x_accel"] = xacc

        return CartPoleState([x, x_dot, theta, sin, cos, theta_dot, force], 0.0, terminal)
    
    def notify_episode_stops(self) -> bool:
        self.reset()
        return True
    
    def reset(self):
        self.x_goal = 0.0
        self.x_start = random.random() - 0.5  # 0.0  # random.random() * 3.4 - 1.7
        # If doing the sway control task, do not spawn near the middle
        x_dot = 0
        theta = np.pi - (random.random() * 0.1 - 0.05)
        theta_dot = 0
        sin = math.sin(theta)
        cos = math.cos(theta)

        state = [self.x_start, x_dot, theta, sin, cos, theta_dot, 0.0]
        cost = self._cost_function(state)

        self._current_state = CartPoleState(state, cost, False)# zero action
        return self._current_state

#class CartPoleUnboundedContinuousEnv(CartPoleUnboundedEnv):
#    continuous = True