# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Cartpole Gym Environments

OpenAI has developed a Cartpole environment with rendering capabilities
for the training of reinforcement learning agents in order to learn how to balance
a pole. We extend this environment to create different tasks involving the cart and
pole, for instance, sway control or swing up.

When creating a new environment, it needs to be registered in envs.__init__.py and
a plant needs to be created which uses that environment.
"""

import math
import random
from typing import Callable, Optional, Tuple, Union

import numpy as np
from gymnasium import logger, spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

from psipy.rl.plants.gym.envs.renderer import RenderMixin, RenderMixinTimeOnly

FLOATMAX = np.finfo(np.float32).max
PRECISION = 6
USE_FRICTION = True


class CartPoleV2Env(RenderMixinTimeOnly, CartPoleEnv):
    """The same exact environment, but with the renderer mixed in."""

    def __init__(self):
        RenderMixinTimeOnly.__init__(self)
        CartPoleEnv.__init__(self)


def polespeed(second, first):
    return np.tan(second-first)

class CartPoleUnboundedEnv(RenderMixin, CartPoleEnv):
    """Cartpole with no angle terminals, allowing the pole to move freely.

    Environment with a pole on a cart. Can be used for both a sway goal or
    a swing up goal, or something else. Behavior depends on the cost function
    used.

        * Sway: cart should be moved to a specific static goal position during which
            the pole should be steadily facing downwards.
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
        cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        sway_control: bool = False,
    ):
        RenderMixin.__init__(self, cost_func)
        CartPoleEnv.__init__(self)
        self.sway_control = sway_control

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
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-np.inf]), np.array([np.inf]), dtype=np.float32
            )
        else:
            # 999 is a placeholder to avoid having to pass in a value
            self.action_space = spaces.Discrete(999)

        self.current_cost = 0
        self.total_cost = 0
        self.current_force = 0.0
        self.step_counter = 0
        self.x_goal = None
        self.x_start = None
        self.start_state = None

        self.reset()

    def step(self, action: Union[float, int]):
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
        info = {}
        x, x_dot, theta, theta_dot, sintheta, costheta = self.state

        # Discrete actions should use the actual force values desired in the action class, instead
        # of just directional -1,0,1 values.
        # if action == 0:
        #     force = 10
        # elif action == 1:
        #     force = 0
        # elif action == 2:
        #     force = -10
        force = action
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

        self.state = (x, x_dot, theta, theta_dot, sin, cos)

        # Cart out of bounds terminal
        done = abs(x) > 2.4
        cost = self.get_state_cost(self.state)
        assert 0 <= cost <= 1
        # Penalize strongly going off screen or high pole
        if done:
            cost = 1
        self.current_cost = cost
        self.total_cost += cost

        if done and self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        elif done:
            assert isinstance(self.steps_beyond_done, int)
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has "
                    "already returned done = True. You should always call "
                    "'reset()' once you receive 'done = True' -- any further "
                    "steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        self.step_counter += 1

        info["force"] = force
        info["theta_accel"] = thetaacc
        info["x_accel"] = xacc

        # The state returned only includes x, x_dot, sin, cos, theta_dot
        state = np.asarray([self.state[0], self.state[1], sin, cos, self.state[3]])
        return state, cost, done, False, info

    def get_state_cost(self, state):
        return 1  # unused, please train on a specific cost function!

    def reset(self):
        self.x_goal = 0.0
        self.x_start = random.random() - 0.5  # 0.0  # random.random() * 3.4 - 1.7
        # If doing the sway control task, do not spawn near the middle
        if self.sway_control:
            # Left or right of center?
            if random.random() < 0.5:
                # Left of center
                self.x_start = random.random() - 2  # [-2,-1)
            else:
                # Right of center
                self.x_start = random.random() + 1  # [1,2)
        x_dot = 0
        theta = np.pi - (random.random() * 0.1 - 0.05)
        theta_dot = 0
        sin = math.sin(theta)
        cos = math.cos(theta)
        self.start_state = (self.x_start, x_dot, theta, theta_dot, sin, cos)
        self.state = (self.x_start, x_dot, theta, theta_dot, sin, cos)
        self.steps_beyond_done = None
        self.step_counter = 0
        self.current_cost = self.get_state_cost(self.state)
        self.total_cost = self.current_cost
        # The state returned only includes x, x_dot, sin, cos, theta_dot
        state = np.asarray((self.x_start, x_dot, sin, cos, theta_dot))
        return np.array(state)


class CartPoleUnboundedContinuousEnv(CartPoleUnboundedEnv):
    continuous = True


class CartPoleAssistedBalanceEnv(RenderMixin, CartPoleEnv):
    """Cartpole but with wider angle limits and fake stops."""

    def __init__(self, cost_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        RenderMixin.__init__(self, cost_func)
        CartPoleEnv.__init__(self)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.0001
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.033  # seconds between state updates
        self.kinematics_integrator = "semieuler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 41 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Goal state definition
        self.x_dot_goal = 0
        self.theta_goal = 0
        self.theta_dot_goal = 0
        self.max_x_dot_change = 0.3

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.current_force = 0
        self.step_counter = 0
        self.start_state = None

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        if self.start_state is None:
            self.start_state = self.state
        force = action
        self.current_force = force
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = (
            force / self.total_mass
        )  # temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        done = bool(x < -self.x_threshold or x > self.x_threshold)

        # Add invisible stops for the pole
        if theta >= self.theta_threshold_radians:
            theta = self.theta_threshold_radians - 0.001
            theta_dot = -1  # add a bounce
        elif theta <= -self.theta_threshold_radians:
            theta = -self.theta_threshold_radians + 0.001
            theta_dot = 1  # add a bounce

        self.state = (x, x_dot, theta, theta_dot)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.step_counter += 1

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # Make pole lie on one of the stops at start.
        self.state[2] = -self.theta_threshold_radians
        if np.random.random() >= 0.5:
            self.state[2] = self.theta_threshold_radians
        self.steps_beyond_done = None
        return np.array(self.state)
