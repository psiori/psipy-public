import math
import random
from typing import Callable, Optional, Tuple, Type
from matplotlib import pyplot as plt
import numpy as np
from psipy.rl.core.plant import Action, Plant, State
from psipy.rl.controllers.nfq import tanh2
from psipy.rl.io.batch import Episode

# This plant contains modified code from the gymnasium cartpole example.
# To satisfy their licence conditions, we include the following statement
# from gymnaisum:

# The MIT License

# Copyright (c) 2016 OpenAI
# Copyright (c) 2022 Farama Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


# The derived work (this file) is also licensed under the BSD license as is
# psipy-public as a whole. See LICENSE file in main directory.   

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

    dtype = "discrete"
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

def make_default_cost_function(x_threshold: float = 2.4) -> Callable[[np.ndarray], np.ndarray]:
    def cost_function(state: np.ndarray) -> np.ndarray:
        x, x_dot, theta, sintheta, costheta, theta_dot, move_ACT = state

        cost = tanh2(theta, C=0.1, mu=0.05) / 10.0

        if (abs(x) >= x_threshold):
            cost = 1.0

        return cost

    return cost_function


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
        x_threshold: float = 2.4,
        cost_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        state_type: Type[CartPoleState] = CartPoleState,
        action_type: Type[CartPoleAction] = CartPoleBangAction,
        render_mode: str = "human",
    ):
        if cost_function is None:
            cost_function = make_default_cost_function(x_threshold)
            print("CartPole is using default cost function")
      
        super().__init__(cost_function=cost_function)

        self.renderable = True

        self.x_threshold = x_threshold
        self.state_type = state_type
        self.action_type = action_type

        self.render_mode = render_mode

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
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400

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
        if abs(x) > self.x_threshold:
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
        self._current_state = CartPoleState(state, 0.0, False)# zero action
        self._current_state.cost = self._cost_function(self._current_state)

        return self._current_state
    
    def render(self):
        #if self.render_mode is None:
        #    print(
        #        "You are calling render method without specifying any render mode. "
        #        "You can specify the render_mode at initialization, "
        #        f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
        #    )
        #    return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed'
            ) from e

        if self.screen is None:
            pygame.init()
            if True: # self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self._current_state is None:
            return None

        x = self._current_state.as_array()

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(100) # self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))



def plot_swingup_state_history(
    episode: Optional[Episode],
    filename: Optional[str] = None,
    episode_num: Optional[int] = None,
    figure: Optional[plt.Figure] = None,
    do_display=True,
    title_string=None
) -> None:
    """Creates a plot that details the controller behavior.

    The plot contains 3 subplots:

    1. Cart position
    2. Pole angle, with green background denoting the motor being active
    3. Action from the controller.  Actions that fall within the red
        band are too small to change velocity, and so the cart does not
        move in this zone.

    Args:
        plant: The plant currently being evaluated, will plot after
                the episode is finished.
        sart_path: If given, will load a sart file instead of requiring
                the plant to run

    """
    cost = None
    #plant = cast(CartPole, plant)

    x = episode.observations[:, 0]
    x_s = episode.observations[:, 1]
    t = episode.observations[:, 2]
    pole_sine = episode.observations[:, 3]
    pole_cosine = episode.observations[:, 4]
    td = episode.observations[:, 5]
    a = episode._actions[:, 0]
    cost = episode.costs
    

    if figure is None:  
        figure, axes = plt.subplots(5, figsize=(10, 8))
    else:
        plt.figure(figure.number)
        axes = figure.axes

    for ax in axes:
        ax.clear()

    axes[0].plot(x, label="cart_position")
    axes[0].set_title("cart_position")
    axes[0].set_ylabel("Position")
    axes[0].legend()

    axes[1].plot(pole_cosine, label="cos")
    axes[1].plot(pole_sine, label="sin")
    axes[1].axhline(0, color="grey", linestyle=":", label="target")
    axes[1].set_title("Angle")
#   axes[1].set_ylim((-1.0, 1,0))
    #axes[1].set_ylabel("Angle")
    axes[1].legend()

    axes[2].plot(td, label="pole_velocity")
    axes[2].set_title("pole_velocity")
    axes[2].set_ylabel("Angular Vel")
    axes[2].legend()

    axes[3].plot(a, label="Action")
    axes[3].axhline(0, color="grey", linestyle=":")
    axes[3].set_title("Control")
    axes[3].set_ylabel("Velocity")
    axes[3].legend(loc="upper left")
 #   axes2b = axs[3].twinx()
 #   axes2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
 #   axes2b.set_ylabel("Steps/s")
 #   axes2b.legend(loc="upper right")

    if cost is not None:
        axes[4].plot(cost, label="cost")
        axes[4].set_title("cost")
        axes[4].set_ylabel("cost")
        axes[4].legend()

    if episode_num is None:
        title = "Simulated Cartpole"
    else:
        title = "Simulated Cartpole, Episode {}".format(episode_num)

    if title_string:
        title = title + " - " + title_string

    figure.suptitle(title)
   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        figure.savefig(filename)
        # plt.close(figure)
    
    if do_display:
        plt.pause(0.01)
    else:
        plt.close()
        return None

    return figure