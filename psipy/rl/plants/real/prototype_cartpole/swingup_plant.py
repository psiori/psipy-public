# PSIORI Reinforcement Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plant that connects via PLC to the swingup hardware in Hamburg.

The Swingup hardware is controlled via a Siemens PLC.  In order to communicate
with it via python, the library `snap7` is used.
"""

import logging
import time
from collections import OrderedDict
from typing import ClassVar, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zmq

from psipy.rl import CM
from psipy.rl.io.logs import log_for_script
from psipy.rl.io.sart import SARTReader
from psipy.rl.plant import Action, Plant, State
from psipy.rl.plant.plc_zmq import Commands, HardwareComms

__all__ = [
    "SwingupAction",
    "SwingupPlant",
    "SwingupState",
]

LOG = logging.getLogger(__name__)
log_for_script("Placeholder", 1, disable_file=True)


def angle_slope(a, b):
    diff = a - b
    sign = np.sign(diff)
    if sign < 0:
        diff *= -1
    return sign * min(diff, 2 * np.pi - diff)


class SwingupAction(Action):
    dtype = "discrete"
    channels = ("direction",)
    legal_values = ((-1, 1, 0, 2, -2),)
    #legal_values = ((-1, 0, 1),)
    #legal_values = ((-1, 1),)
    #legal_values = ((-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1),)

class SwingupDiscretizedAction(Action):
    dtype = "discrete"
    channels = ("direction",)
    legal_values = ((-1,1,0,2,-2),)

class SwingupContinuouDiscreteAction(Action):
    dtype = "discrete"
    channels = ("direction",)
    legal_values = ((-100,100,0,400,-400,800,-800,1000,-1000,200,-200,20,-20,600,-600),)

class SwingupTwoAct(Action):
    dtype = "discrete"
    channels = ("direction",)
    legal_values = ((-1,1),)


class SwingupState(State):
    _channels = (
        "cart_position",
        "cart_velocity",
        "pole_angle",
        "pole_velocity",
        "direction_ACT",
    )


class SwingupPlant(Plant[SwingupState, SwingupContinuouDiscreteAction]):
    """Hardware plant for Cartpole Swingup.

    This plant connects to the hardware via a Siemens PLC.
    Max cycle time is close to 30 Hz but not quite.  Be wary of setting
    the ACTION_DELAY too low.

    To connect via a Mac, change the Network Config of the USB Connection to:

    - Configure: Manually
    - IP: 192.168.0.10 (10 can be anything but not 1 or 2)
    - Mask: 255.255.255.0

    Args:
        plc_address: PLC IP address as a string  #TODO: Update
        speed_value: The speed in either direction to be used by the cart
        plot: Whether or not to plot the episode after it is completed.
        reposition_on_start: Move back to the center when starting a new episode
            if calibration did not occur.
        block_swinging: Wait until the pole stopped swinging before activating the cart.

    """

    state_type = SwingupState
    action_type = SwingupContinuouDiscreteAction
    renderable = False

    # Axis specific dimensions
    LEFT_SIDE: int = 0
    RIGHT_SIDE: int = 6755
    CENTER: int = (RIGHT_SIDE - LEFT_SIDE) // 2
    TICKS_PER_CIRCLE: int = 600
    TERMINAL_LEFT_OFFSET: int = 100
    TERMINAL_RIGHT_OFFSET: int = 300

    # Minimum time to wait between performing an action and receiving a
    # consecutive state. Allows the physical model to actually perform a
    # movement in reaction to a newly set velocity.
    ACTION_DELAY: ClassVar[float] = 0.020

    def __init__(
        self,
        port: str,
        speed_value: int,
        speed_value2: Optional[int] = None,
        plot: bool = False,
        reposition_on_start: bool = True,
        block_swinging: bool = False,
        balance_task:bool = False,
        continuous:bool = False,
    ):
        super().__init__()
        self.comms = HardwareComms(port)
        LOG.info("Connected to Hilscher.")
        self.comms.send(Commands.RESET_ALL)
        self.comms.receive()
        self.comms.send(Commands.HERTZ, self.ACTION_DELAY * 1000)
        self.comms.receive()

        # Plant behavior
        self.speed = speed_value
        #self.set_speed(speed_value)
        self.reposition_on_start = reposition_on_start
        self._calibrated = False
        self.swing_block = block_swinging
        self.startup = True
        self.discretized = False
        if speed_value2 is not None:
            LOG.warning("Running plant in Discretized Mode!")
            self.discretized = True
            assert speed_value2 is not None
            self.speed2 = speed_value2
            self.set_speed2(speed_value2)
        self.continuous = continuous
        # if len(self.action_type.legal_values[0]) > 3:
        #     LOG.info("Using SwingupPlant in discretized continuous mode!")
        #     self.continuous = True
        # Task specific
        self.balance_task: bool = balance_task

        # Analysis
        self.plot = plot
        self._solved = False
        self.df_history = pd.DataFrame(
            columns=[
                "cart_position",
                "cart_velocity",
                "pole_angle",
                "pole_velocity",
                "direction",
            ]
        )

    def calculate_cycle_time(self, n_obs: int):
        """Calculate real plant time for n number of observations."""
        return n_obs * self.ACTION_DELAY

    def convert_angle(self, angle: int) -> int:
        """Convert angles to and from PLC units.

        The PLC sends angles in a range between [0,600]. This function
        converts those angles to and from the [-pi, pi] range, with 0
        being up vertically.
        """
        angle = angle - ((angle // self.TICKS_PER_CIRCLE) * self.TICKS_PER_CIRCLE)
        angle = np.pi / (self.TICKS_PER_CIRCLE / 2) * angle - np.pi
        return angle

    def read_state(self) -> Tuple[int, int, float, bool]:
        """Read the current state from the plant.

        Hardware stops are True when not triggered, so we flip them here.
        Angles are in a range [0, 600].  These are later converted to radians
        with 0 being vertical.

        Returns:
            state as (position, angle, timestep, left stop, raw cart vals)
        """
        ts = time.time()  # basically instantaneous
        state = self.comms.receive()
        state = state.split(" ")
        x = int(state[0])
        theta = int(state[1])
        lstop = bool(state[2])

        return x, theta, ts, lstop

    def reset(self):
        """Reset the PLC's values and clear calibration."""
        self.comms.send(Commands.RESET_ALL)
        self.comms.receive()
        self.set_speed(self.speed)
        self._calibrated = False

    def set_speed(self, speed: int):
        """Set the velocities for all three movement commands."""
        self.comms.send(Commands.SET_SPEED, speed)
        self.comms.receive()

    def set_speed2(self, speed:int):
        """Set the velocites for the second set of speeds."""
        self.comms.send(Commands.SET_SPEED2, self.speed2)
        self.comms.receive()

    def go_right(self, speed=None):
        # if not self.continuous:
        self.comms.send(Commands.EXEC_SPEED2)
        # else:
        #     if speed is None:
        #         speed = self.speed
        #     self.comms.send(Commands.RIGHT_CONT, speed)

    def go_left(self, speed=None):
        # if not self.continuous:
        self.comms.send(Commands.EXEC_SPEED0)
        # else:
        #     if speed is None:
        #         speed = self.speed
        #     self.comms.send(Commands.LEFT_CONT, speed)

    def go_left2(self, speed=None):
        # if not self.continuous:
        self.comms.send(Commands.EXEC_SPEED3)
        # else:
        #     raise NotImplementedError

    def go_right2(self, speed=None):
        # if not self.continuous:
        self.comms.send(Commands.EXEC_SPEED4)
        # else:
        #     raise NotImplementedError

    def halt(self):
        self.comms.send(Commands.EXEC_SPEED1)

    def motor_on(self) -> None:
        """Turn the motor on."""
        self.comms.send(Commands.MOTOR_ON)
        self.comms.receive()
        LOG.info("Motor turned on.")

    def motor_off(self) -> None:
        """Turn the motor off"""
        self.comms.send(Commands.MOTOR_OFF)
        self.comms.receive()
        LOG.info("Motor turned off.")

    def calibrate(self) -> None:
        """Activate the homing calibration of the PLC."""
        LOG.info("Calibrating...")
        self.comms.send(Commands.CALIBRATE)
        self.comms.receive()
        LOG.info("Calibrated.")
        self._calibrated = True

    def return_to_center(self) -> None:
        """Returns the cart to the center of the axis."""
        LOG.info("Returning to center...")
        if not self.calibrated:
            LOG.warning("Trying to go to center of an uncalibrated axis!")
            raise RuntimeWarning
        self.comms.send(Commands.CENTER)
        self.comms.receive()
        LOG.info("Returned to center.")

    def check_initial_state(self, state: Optional[SwingupState]) -> SwingupState:
        assert self.episode_steps == 0
        self.comms.send_NOOP()  # need to send to receive
        x, theta, ts, _ = self.read_state()
        time.sleep(self.ACTION_DELAY)
        self.comms.send_NOOP()
        x_, theta_, ts_, _ = self.read_state()
        x_dot = x_ - x
        theta = self.convert_angle(theta)
        theta_ = self.convert_angle(theta_)
        theta_dot = angle_slope(theta, theta_)
        obs = OrderedDict(
            cart_position=x_,
            cart_velocity=x_dot,
            pole_angle=theta_,
            pole_velocity=theta_dot,
            direction_ACT=0.0,
        )
        self._current_state = self.state_type(obs, meta=dict(ts=ts_))
        self._record_history(*obs.values())
        return self._current_state

    def _get_next_state(
        self, state: SwingupState, action: SwingupAction
    ) -> SwingupState:
        assert state == self._current_state
        direction = action["direction"]

        # a = time.time()
        with CM["get-state/vel"]:
            # Get previous measurements to compute slopes on later.
            if self.continuous:
                self.set_speed(direction)
                self.go_right()
            else:
                if not self.discretized:
                    if direction == 1:
                        self.go_right()
                    elif direction == -1:
                        self.go_left()
                    else:
                        self.halt()
                else:
                    if direction == 1:
                        self.go_right()
                    elif direction == -1:
                        self.go_left()
                    elif direction == 2:
                        self.go_right2()
                    elif direction == -2:
                        self.go_left2()
                    else:
                        self.halt()
                # if direction <= 1 and direction > 0:
                #     self.go_right(self.speed * abs(direction))
                # elif direction >= -1 and direction < 0:
                #     self.go_left(self.speed * abs(direction))
                # else:
                #     self.halt()

        x = float(self._current_state["cart_position"])
        theta = cast(int, self._current_state["pole_angle"])
        ts = cast(float, self._current_state.meta["ts"])

        with CM["get-state/readstate"]:
            # Build new state given new readings.
            x_, theta_, ts_, _ = self.read_state()
            if x_> 10000:
                LOG.warning("Overflow in position!")
                x_ = 0  # prevent overflow
            x_dot = x_ - x
            theta_ = self.convert_angle(theta_)
            theta_dot = angle_slope(theta_, theta)
            obs = dict(
                cart_position=x_,
                cart_velocity=x_dot,
                pole_angle=theta_,
                pole_velocity=theta_dot,
                direction_ACT=direction,
            )

        # Determine if terminal state
        bad_position = not (
                self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET
                <= x_
                <= self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET)
        if self.balance_task:
            bad_angle = np.abs(theta_) > .78
            print("there", theta_, bad_angle)
            terminal = bad_angle or bad_position
        else:
            terminal = bad_position

        # Stop the cart if in terminal state
        if terminal:
            self.halt()
            LOG.info(f"Stopped the cart due to termination ({round(x_,2)}, {round(theta_,2)}).")

        next_state = self.state_type(obs, terminal=terminal, meta=dict(ts=ts_))
        self._record_history(*obs.values())
        self._current_state = next_state
        return self._current_state

    def enforce_pole_down(self, consecutive_steps: int = 5) -> None:
        """Blocks until pole stops moving and hangs downwards.

        Args:
            consecutive_steps: How many 1/10s of a second to have the same pole
                               angle before saying it is stopped.
        """
        stopped = 0
        self.comms.send(Commands.NOOP)
        _, theta, _, _ = self.read_state()
        LOG.info("Waiting until pole is stable...")
        while stopped < consecutive_steps:  # Requires int angles
            time.sleep(0.1)
            self.comms.send(Commands.NOOP)
            _, theta_, _, _ = self.read_state()
            if abs(theta_ - theta) < 50:
                stopped += 1
            else:
                # Reset angle and counter
                theta = theta_
                stopped = 0
        LOG.info("Pole stabilized.")

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_solved(self) -> bool:
        return self._solved

    def _record_history(
        self, x_: float, x_dot: float, theta_: float, theta_dot: float, vel: float
    ) -> None:
        """Record a state in the local history for analysis"""
        self.df_history.loc[self.episode_steps, "cart_position"] = x_
        self.df_history.loc[self.episode_steps, "cart_velocity"] = x_dot
        self.df_history.loc[self.episode_steps, "pole_angle"] = theta_
        self.df_history.loc[self.episode_steps, "pole_velocity"] = theta_dot
        self.df_history.loc[self.episode_steps, "velocity"] = vel

    def solve_condition(self, state: SwingupState):
        """Return True if potentially solved.

        If angle < .15 rad (~8 deg) off vertical at end of episode,
        the task is potentially solved.  It may have just ended within
        the goal region, and so the saved model needs to be double checked.
        """
        angle_goal = .15 if not self.balance_task else .05
        if np.abs(state["pole_angle"]) < angle_goal:
            return True
        return False

    def get_ready(self):
        # Only calibrate if we aren't already calibrated, to save time
        if not self.calibrated:
            self.calibrate()
        self.return_to_center()

    def notify_episode_starts(self) -> bool:
        super().notify_episode_starts()
        if self.startup:
            self.motor_on()
            self.get_ready()
        if self.swing_block:
            self.enforce_pole_down()
        return True

    def notify_episode_stops(self) -> bool:
        time.sleep(0.1)
        try:
            self.comms.receive()
        except zmq.error.ZMQError:
            pass
        self.halt()
        self.comms.receive()
        self.motor_off()
        self.reset()  # was already calibrated before
        self._solved = self.solve_condition(self._current_state)
        if self.plot:
            plot_swingup_state_history(self)
        self.startup = False
        self.motor_on()
        self.get_ready()
        return True

    def __del__(self):
        self.halt()
        self.comms.receive()
        self.motor_off()


def plot_swingup_state_history(
    plant: Optional[SwingupPlant] = None,
    filename: Optional[str] = None,
    sart_path: Optional[str] = None,
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
    if sart_path:
        with SARTReader(sart_path) as sart:
            sart = sart.load_full_episode()
            x = sart[0][:, 0]
            x_s = sart[0][:, 1]
            t = sart[0][:, 2]
            a = sart[0][:, 4]
    else:
        plant = cast(SwingupPlant, plant)
        x = plant.df_history.cart_position
        x_s = plant.df_history.cart_velocity
        t = plant.df_history.pole_angle
        a = plant.df_history.direction

    fig, axs = plt.subplots(3, figsize=(10, 5))
    axs[0].plot(x, label="cart_position")
    axs[0].set_title("cart_position")
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(t, label="pole_angle")
    axs[1].axhline(0, color="grey", linestyle=":", label="target")
    axs[1].set_title("Pole Angle")
    axs[1].set_ylim((-3.15, 3.15))
    axs[1].set_ylabel("Angle")
    axs[1].legend()

    axs[2].plot(a, label="Action")
    axs[2].axhline(0, color="grey", linestyle=":")
    axs[2].set_title("Control")
    axs[2].set_ylabel("Velocity")
    axs[2].legend(loc="upper left")
    axs2b = axs[2].twinx()
    axs2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
    axs2b.set_ylabel("Steps/s")
    axs2b.legend(loc="upper right")

    plt.suptitle("NFQ Controller on Physical Swingup Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    # sart_path = (
    #     "../../../examples/rl/mon-sart-hardware-swingup/"
    #     "sart-Hardware Swingup-37-20200203-104033.hdf5"
    # )
    # plot_swingup_state_history(None, sart_path=sart_path)
    plant = SwingupPlant("5555", speed_value=100, continuous=True)
    plant.calibrate()
    plant.return_to_center()
    plant.set_speed(100)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.set_speed(-100)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.set_speed(800)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.set_speed(-800)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.set_speed(0)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.motor_off()
    exit()


    plant.go_left()
    plant.comms.receive()
    time.sleep(1)
    plant.go_right()
    plant.comms.receive()
    time.sleep(1)
    plant.halt()
    plant.comms.receive()
    time.sleep(1)
    plant.go_left2()
    plant.comms.receive()
    time.sleep(1)
    plant.go_right2()
    plant.comms.receive()
    time.sleep(1)
    plant.halt()
    plant.comms.receive()
    plant.reset()
    exit(0)

    times = []
    for _ in range(100):
        times.append(time.time())
        plant.read_state()
    times = np.array(times)
    times = times[1:] - times[:-1]
    print("read_state timings:", times.mean(), "+-", times.std())

    # plant.motor_off()
    # exit()
    plant.motor_on()
    plant.calibrate()
    plant.return_to_center()
    try:
        plant.motor_off()
        while True:
            x = plant.read_state()
            print("X", x[0], "T", plant.convert_angle(x[1]))
            time.sleep(0.2)
    except KeyboardInterrupt:
        plant.motor_off()
