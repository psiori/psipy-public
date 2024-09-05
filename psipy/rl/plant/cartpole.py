# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plant that connects to the Freiburg sway model via Arduino.

Included also are NFQCA based Sway controllers and a Sway explorer
to collect randomized data.

TODO: More extensive explanation of how for example communication works?

"""

import logging
import os
import sys
import time
from typing import ClassVar, Optional, Tuple, Type, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from serial import Serial

from psipy.core.threading_utils import StoppableThread
from psipy.rl.control import Controller
from psipy.rl.control.nfqca import NFQCA
from psipy.rl.plant import Action, Plant, State
from psipy.rl.plant.imu import IMUConnection

__all__ = [
    "CartPoleAction",
    "CartPolePlant",
    "CartPoleState",
    "SwayExplorer",
    "SwayNFQCA",
]


LOG = logging.getLogger(__name__)
BAUDRATE = 500000
LEFT_SIDE = -500
RIGHT_SIDE = 6500
TTY_PREFIX = "tty.usbmodem"
if sys.platform == "linux":
    # TODO(axelperschmann): Check if this is always correct on linux
    TTY_PREFIX = "ttyACM0"


def connect_serial(tty: str) -> Serial:
    if not os.path.exists(tty):
        try:
            name = next(name for name in os.listdir("/dev") if name.startswith(tty))
        except StopIteration:
            raise ValueError(f"Could not connect to {tty}")
        tty = os.path.join("/dev", name)
    try:
        device = Serial(tty, timeout=1, baudrate=BAUDRATE)
    except OSError:
        device = Serial(tty, timeout=1)
    LOG.info(f"Connected to {tty}")
    return device


class ArduinoComm(StoppableThread):
    def __init__(self, tty):
        super().__init__(daemon=True)
        self.serial = connect_serial(tty)
        self._data = (0, 0, 0, 0)
        self._tock = 0

    def __del__(self):
        self.serial.close()

    def get(self):
        return self._data

    def write(self, msg):
        self.serial.write(msg)

    def run(self):
        """Read most recent line from Arduinos SerialBuffer."""
        buffer = ""
        while not self.stopped():
            rec = ""
            buffer_len = 0
            while not self.stopped():
                try:
                    if self.serial.in_waiting > 0:
                        buffer += self.serial.read_all().decode("utf-8")
                    else:  # block till one line is readable or timeout occurs
                        buffer += self.serial.readline().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if buffer_len == len(buffer):
                    raise ValueError("Timeout, did not receive any data from Arduino.")
                index = buffer.rfind("\r\n")
                if index == -1:
                    continue
                rec, buffer = buffer[:index], buffer[index:]
                try:
                    rec = next(line for line in rec.split("\r\n")[::-1] if line)
                    x, theta, ts = np.fromstring(rec, sep=",", dtype=int)
                except (StopIteration, ValueError, NameError):
                    continue
                break
            theta = 360 / 4096.0 * theta  # Convert theta to degrees
            theta = (theta - 180) % 360  # 180° pointing downwards
            theta = theta - 180  # 0° pointing downwards
            ts = ts * 1e-6  # micros to seconds
            self._data = (x, theta, ts, time.time())


class CartPoleAction(Action):
    dtype = "continuous"
    channels = ("velocity",)
    legal_values = ((-1, 1),)  # [Lower, Upper]


class CartPoleState(State):
    _channels = (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_velocity",
        "ax",
        "ay",
        "gz",
        "velocity_ACT",
    )


class CartPolePlant(Plant[CartPoleState, CartPoleAction]):
    state_type = CartPoleState
    action_type = CartPoleAction
    meta_keys = ("ts", "goal")

    #: Minimum time to wait between performing an action and receiving a
    #: consecutive state. Allows the physical model to actually perform a
    #: movement in reaction to a newly set velocity.
    ACTION_DELAY: ClassVar[float] = 0.02  # IMU frequency

    MIN_PERC = 0.06976
    PULSE_MIN: ClassVar[int] = 300
    PULSE_MAX: ClassVar[int] = 4300
    STEPSIZE: ClassVar[int] = 40

    _tick: float

    def __init__(
        self,
        tty: str = TTY_PREFIX,
        plot: bool = False,
        reset_on_start: bool = False,
        use_imu: bool = False,
    ):
        super().__init__()
        self.buffer = ""
        self.comm = ArduinoComm(tty)
        self.comm.start()
        self.use_imu = use_imu
        self.imu: Optional[IMUConnection] = None
        if self.use_imu:
            self.imu = IMUConnection("192.168.178.95")
            self.imu.start()
        self.plot = plot
        self.reset_on_start = reset_on_start

    def _convert_percent_to_vel(self, percentage: float) -> int:
        """Convert percentage to an integer velocity which is understood by the arduino.

        Args:
            percentage: float, range -1. to +1.
        """
        ZEROVEL = 128
        if np.isclose(percentage, 0):
            return ZEROVEL

        percentage = np.clip(percentage, -1.0, 1.0)
        sign = np.sign(percentage)
        percentage = abs(percentage)

        # Warn about pulse width OOB, and clip
        pulse_width = int(self.PULSE_MIN / percentage)
        if pulse_width > self.PULSE_MAX:
            LOG.warning(
                f"Warning! Baudrate above maximum: {pulse_width} > {self.PULSE_MAX}"
            )
        pulse_width = np.clip(pulse_width, a_min=self.PULSE_MIN, a_max=self.PULSE_MAX)

        # Convert to int
        vel = 100 - (pulse_width - 300) / self.STEPSIZE
        return int(vel * sign) + ZEROVEL

    def set_velocity(self, vel):
        self.comm.write(self._convert_percent_to_vel(vel).to_bytes(1, "big"))

    def set_position(self, pos: float = LEFT_SIDE, precision=3, block=False):
        """Brings the pole to the given position as quickly as possible."""

        def _goal_pos(pos, x, direction):
            if direction == 1:
                return pos - x
            return x - pos

        pos = np.clip(pos, 100, 6500)
        x, theta, _, _ = self.receive()
        direction = np.sign(pos - x)
        vel = np.clip(_goal_pos(pos, x, direction), -1, 1)
        while _goal_pos(pos, x, direction) > 0:
            x, theta, _, ts = self.receive()
            self.set_velocity(vel * direction)
        self.set_velocity(0)

        swing_vel = 999.0
        while block and (abs(theta) > precision or swing_vel != 0):
            raise NotImplementedError
            # Breaks when angle AND swing speed indicate steady pole
            # _, theta, _, ts = self.receive()
            # time.sleep(0.1)
            # _, theta_, _, ts_= self.receive()
            # swing_vel = calculate_slope((theta, ts), (theta_, ts_))

    def notify_episode_starts(self):
        """Starts plant and calibrates left side of track as 0 position"""
        super().notify_episode_starts()
        if self.reset_on_start:
            LOG.info("Calibrating...")
            self.calibration = 0
            x, _, _, _ = self.receive()
            self.set_velocity(-1)
            while True:
                time.sleep(0.2)
                x_, _, _, _ = self.receive()
                if x_ == x:
                    break
                x = x_
            self.calibration = x_
            self.set_position(3500)
            LOG.info("Moved to center.")
            time.sleep(1)
            LOG.info("Calibration complete.")
        self.df_history = pd.DataFrame(
            columns=[
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_velocity",
                "velocity",
            ]
        )
        return True

    def check_initial_state(
        self, state: Optional[CartPoleState] = None
    ) -> CartPoleState:
        assert self.episode_steps == 0
        # Read two consecutive states to have initial velocity measurements.
        x, theta, _, ts = self.receive()
        time.sleep(self.ACTION_DELAY)
        x_, theta_, raw_, ts_ = self.receive()
        x_dot = calculate_slope((x, ts), (x_, ts_))
        theta_dot = calculate_slope((theta, ts), (theta_, ts_))

        obs = dict(
            cart_position=x_,
            cart_velocity=x_dot,
            pole_angle=theta_,
            pole_velocity=theta_dot,
            velocity_ACT=0.0,
        )
        if raw_:
            obs["ax"], obs["ay"], obs["gz"] = raw_

        self._current_state = self.state_type(obs, meta=dict(ts=ts_, goal=None))
        return self._current_state

    def _get_next_state(
        self, state: CartPoleState, action: CartPoleAction
    ) -> CartPoleState:
        # print(state, self._current_state)
        # TODO: The following assert should be added back at some point. It
        # currently is disabled because for sway control the state is mutated
        # before it is written to the sart files to have it represent the goal.
        # assert state == self._current_state
        vel = action["velocity"]
        assert -1.0 <= vel <= 1.0

        # Get previous measurements to compute slopes later on.
        x = float(self._current_state["cart_position"])
        theta = float(self._current_state["pole_angle"])
        ts = cast(float, self._current_state.meta["ts"])

        self.set_velocity(vel)

        # Sleep to give hardware time to do something
        # time.sleep(self.ACTION_DELAY)

        # Sleep to make loop time deterministic
        delta = time.time() - self._tick
        # if delta < self.ACTION_DELAY:
        time.sleep(max(self.ACTION_DELAY - delta, 0.001))

        # Build new state given new readings.
        x_, theta_, raw_, ts_ = self.receive()
        x_dot = calculate_slope((x, ts), (x_, ts_))
        theta_dot = calculate_slope((theta, ts), (theta_, ts_))
        obs = dict(
            cart_position=x_,
            cart_velocity=x_dot,
            pole_angle=theta_,
            pole_velocity=theta_dot,
            velocity_ACT=vel,
        )
        if raw_:
            obs["ax"], obs["ay"], obs["gz"] = raw_
        next_state = self.state_type(obs, meta=dict(ts=ts_, goal=None))

        self.df_history.loc[self.episode_steps, "cart_position"] = x_
        self.df_history.loc[self.episode_steps, "cart_velocity"] = x_dot
        self.df_history.loc[self.episode_steps, "pole_angle"] = theta_
        self.df_history.loc[self.episode_steps, "pole_velocity"] = theta_dot
        self.df_history.loc[self.episode_steps, "velocity"] = vel

        # next_state.terminal = True if x_ == 0 else False
        # next_state.terminal = True if np.abs(theta_ - 180) > 30 else False
        # print(next_state.terminal)
        self.validate_next_state(state, action, next_state)
        self._current_state = next_state
        return self._current_state

    def receive(self):
        x, theta, ts, self._tick = self.comm.get()
        raw = (0.0, 0.0, 0.0)
        if self.use_imu:
            ts_imu, _, pitch, roll, _, pitch_dot, _ = self.imu.get_euler_dot()
            if roll >= 0:
                angle_sign = 1
            else:
                angle_sign = -1
            theta = (pitch + 90) * angle_sign
            raw = (self.imu.ax, self.imu.ay, self.imu.gz)
            # angle_error = (pitch + 90) * angle_sign / 5.0
        return x, theta, raw, ts

    def notify_episode_stops(self):
        self.set_velocity(0)
        if self.reset_on_start:
            self.set_position(0, block=False)
        if self.plot:
            plot_state_history(self)

    def __del__(self):
        # self.set_velocity(0)
        if hasattr(self, "comm"):
            self.comm.stop()
        # self.set_position(3500, block=False)


class SwayPlant:
    """A cart moving between different target positions constantly.

    This plant simulates behaviour where an operator would move the gantry
    between different positions constantly. The goal is to have an active sway
    control system which transforms the operator commands in a way that they
    still behave as expected, but without a swaying pole.
    """


def calculate_slope(obs: Tuple[float, float], obs_: Tuple[float, float]) -> float:
    """Compute slope based on two measurements at two consecutive timesteps."""
    x, t = obs
    x_, t_ = obs_
    x_delta = x_ - x
    t_delta = t_ - t
    # assert t_delta >= 0.001
    if x_delta == 0:
        return 0
    return x_delta / t_delta


class SwayExplorer(Controller):
    def __init__(
        self,
        state_channels: Tuple[str, ...],
        action_type: Type[CartPoleAction],
        num_repeat: int = 3,
        noise: float = 0.0,
    ):
        super().__init__(state_channels, action_type)
        self.num_repeat = num_repeat
        self.repeats = num_repeat
        self.prev = np.zeros(1)
        self.noise = noise
        self.goal = 3000

    def notify_episode_starts(self) -> None:
        print("Episode goal:", self.goal)

    def notify_episode_stops(self) -> None:
        pass

    def get_action(self, state: State) -> Action:
        state.meta["goal"] = self.goal
        state.cost = state["cart_position"] - self.goal
        state["cart_position"] = self.goal - state["cart_position"]  # inplace!
        state_values = state.as_array(*self.state_channels)
        action_values = self._get_action(state_values)
        return self.action_type(action_values)

    def _get_action(self, state: np.ndarray) -> np.ndarray:
        if self.repeats < self.num_repeat:
            self.repeats += 1
            return self.prev
        self.prev = np.clip(np.random.normal(0, 0.5, (1,)), -1, 1)
        # if np.random.random() < self.noise or state[0] >= 0:
        #     self.prev = np.array([float(random.randint(-1, 1))])
        # else:
        #     self.prev = np.array([1.0])
        self.repeats = 0
        return self.prev
        # if state[0] < 0:
        #     return np.array([1.0])


class SwayNFQCA(NFQCA):

    _switch_goals: ClassVar[Tuple[int, int]] = (2000, 4000)

    def __init__(self, *args, goal: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal = goal
        self.previous_goal = 1
        self.current_goal = self._switch_goals[0]

    def notify_episode_starts(self) -> None:
        if self.goal is not None:
            self.current_goal = self.goal
        else:
            self.previous_goal = (self.previous_goal + 1) % 2
            self.current_goal = self._switch_goals[self.previous_goal]
        LOG.info(f"Episode goal: {self.current_goal}")

    def _get_action(self, observation: np.ndarray) -> np.ndarray:
        # observation[0] = observation[0] - self.current_goal

        # In the pure shaping setting, the absolute OR relative cart position is
        # of no relevance. Instead, only the operator's cart position actions
        # and the historic cart movements (given by its velocity) are of
        # interest.
        assert self.state_channels[0] == "cart_position"
        x = observation[0]
        system_action = -1
        if self.current_goal - x > 0:
            system_action = 1
        observation[0] = system_action
        action = super()._get_action(observation)
        LOG.info(f"Operator: {observation[0]:.2f} | Shaped: {action:.2f}")
        return action


def plot_state_history(plant: CartPolePlant, filename: Optional[str] = None) -> None:
    """Creates a plot that details the controller behavior.

    The plot contains 3 subplots:
        1. Cart position
        2. Pole angle, with green background denoting the motor being active
        3. Action from the controller.  Actions that fall within the red
            band are too small to change velocity, and so the cart does not
            move in this zone.
    """
    fig, axs = plt.subplots(3, figsize=(20, 12))
    # axs[0].axhline(6000, label="6000 goal", color="yellow")
    axs[0].plot(plant.df_history.cart_position, label="cart_position")
    axs[0].set_title("cart_position")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(plant.df_history.pole_angle, label="pole_angle")
    axs[1].axhline(0, color="grey", linestyle=":", label="target")
    axs[1].set_title("Pole Angle")
    axs[1].set_ylim((-30, 30))
    axs[1].set_ylabel("Angle")
    axs[1].fill_between(
        plant.df_history.index,
        y1=-30,
        y2=-30 + (abs(plant.df_history.velocity) > plant.MIN_PERC) * 60.0,
        alpha=0.1,
        color="green",
        label="motor active",
    )
    axs[1].legend()

    axs[2].plot(plant.df_history.velocity * 100, label="Action")
    axs[2].axhspan(
        -plant.MIN_PERC * 100,
        plant.MIN_PERC * 100,
        alpha=0.3,
        color="red",
        label=f"Velocity <= {plant.MIN_PERC * 100}% (zero motion zone)",
    )
    axs[2].axhline(0, color="grey", linestyle=":")
    axs[2].set_title("Control")
    axs[2].set_ylabel("Velocity [%]")
    axs[2].yaxis.set_major_formatter(PercentFormatter())
    axs[2].legend(loc="upper left")
    axs2b = axs[2].twinx()
    axs2b.plot(
        plant.df_history.cart_velocity, color="black", alpha=0.4, label="True Velocity"
    )
    axs2b.set_ylabel("Steps/s")
    axs2b.legend(loc="upper right")

    plt.suptitle("NFQCA Controller on Physical CartSwayModel")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename)
    plt.show()
