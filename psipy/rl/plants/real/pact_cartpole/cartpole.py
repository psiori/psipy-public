# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Plant that connects to Swingup hardware in Hamburg.

Connection is made via ZMQ communication to C code which connects to a
Hilscher PROFINET card, which in turn is connected to the PLC on the
Swingup hardware in Hamburg.
"""

import logging
import time
from collections import OrderedDict
from typing import Callable, cast, ClassVar, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zmq

from psipy.rl import CM
from psipy.rl.core.controller import Controller
from psipy.rl.io.sart import SARTReader
from psipy.rl.core.plant import Action, Plant, State

from psipy.rl.plants.real.pact_cartpole.plc_zmq import Commands, HardwareComms

__all__ = [
    "SwingupAction",
    "SwingupPlant",
    "SwingupState",
]

LOG = logging.getLogger(__name__)


# def angle_slope(a, b):
#     diff = a - b
#     sign = np.sign(diff)
#     if sign < 0:
#         diff *= -1
#     return sign * min(diff, 2 * np.pi - diff)


# def angle_slope(a, b):
#     return np.tan(b - a)

def angle_slope(end, start):
    return np.arctan2(np.sin(start - end), np.cos(start - end))


class SwingupAction(Action):
    """Left, right, or stop"""

    dtype = "discrete"
    channels = ("direction",)
    legal_values = ((-1, 0, 1),)


class SwingupDiscretizedAction(Action):
    """Three speeds in either direction."""

    dtype = "discrete"
    channels = ("direction",)
    # legal_values = ((0, 1, 2, 3, 4),)
    legal_values = ((1, 2, 3),)


class SwingupContinuousDiscreteAction(Action):
    """Any speed in either direction."""

    dtype = "discrete"
    channels = ("direction",)
    legal_values = (
        (  # SL: problem: 0 is not zero :-()  present zero is near 150 (oh: due to operator! )
            #150,
            #-100,
            #100,
            #200,
            #-200,
            #300,
            #-300,
            #400,
            #-400,
            # 500,
            #-500,
            # 600,
            # -600,
            # 700,
            -500,
            500,
            #-700,
            # 900,
            # -900,
            # 1000,
            # -1000,
        ),
    )


class SwingupContinuousAction(Action):
    dtype = "continuous"
    channels = ("direction",)
    legal_values = ((-1000, 1000),)


class SwingupState(State):
    _channels = (
        "cart_position",     #TODO: is goal position
        "cart_velocity",
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        "pole_angle",
#        "true_position",
        "dist_left",
        "dist_right",
#       "operator",
        "direction_ACT",
    )


class SwingupPlant(Plant[SwingupState, SwingupContinuousDiscreteAction]):
    """Hardware plant for Cartpole Swingup.

    This plant connects to the hardware via a ZMQ request/reply socket to C code.
    The C code in turn connects to the Siemens PLC via a Hilscher PROFINET card.

    To connect via a Mac, change the Network Config of the USB Connection to:

    - Configure: Manually
    - IP: 192.168.0.10 (10 can be anything but not 1 or 2)
    - Mask: 255.255.255.0

    ZMQ will then connect to the C code on port 5555.

    The plant also subscribes to a shutdown topic from the PACT Config UI.
    When receiving a shutdown command, the python script will shut down gracefully.

    Available speeds of the cart are determined by the passed "speed_values". This can
    be any number of speeds, but need to follow the following criteria:

        * Does not contain 0.
        * Does not contain any negative numbers (these are added automatically).

    Activation of multiple speed values need to be implemented in ``plc_zmq.py`` and in
    the C code. Currently the C code supports 3 speeds, and with inverse and 0, in
    total, 7 speeds are available.

    There is also a continuous mode, where a given speed is written into the speed
    block before executing it. This allows for any desired speed, and the speed
    changes are taken care of C side.

    Args:
        hilscher_port: Port to communicate via ZMQ
        command_port: Port to subscribe to shutdown requests
        speed_values: List of POSITIVE(!) speed values to be used by the cart.
                      The negative versions will be automatically added, as well
                      as 0 speed. For example, [1,2,3] will allow the following
                      speeds: [-3,-2,-1,0,1,2,3].
                      Ignored when in continuous mode.
        plot: Whether or not to plot the episode after it is completed.
        reposition_on_start: Move back to the center when starting a new episode
                             if already calibrated. If a PLC-side soft stop is hit,
                             the cart will recalibrate regardless of this setting.
        block_swinging: Wait until the pole stopped swinging before activating the cart
        angle_terminals: Adds a terminal state based on angle; use when controller
                         should learn the vertical balancing task.
        continuous: Allow setting speeds on the fly C side for many-speed operation
        backward_compatible: Use a faulty continuous speed calculation for models that
                             split the speed setting and execution python side. These
                             models essentially have a half as fast action delay.
        sway_start: Start with the cart offc enter so that it has to move to the center
                       on its own.
        cost_func: If provided, the cost of the current state is sent over ZMQ to the
                   C side.
        controller: If provided, the Q value for the current state/action pair is sent
                    over ZMQ to the C side.
    """

    state_type = SwingupState
    action_type = SwingupContinuousDiscreteAction
    renderable = False

    # Axis specific dimensions
    LEFT_SIDE: int = 0
    RIGHT_SIDE: int = 6755
    CENTER: int = (RIGHT_SIDE - LEFT_SIDE) // 2
    TICKS_PER_CIRCLE: int = 600
    TERMINAL_LEFT_OFFSET: int = 300
    TERMINAL_RIGHT_OFFSET: int = 200

    # Minimum time to wait in seconds between performing an action and receiving
    # a consecutive state. Allows the physical model to actually perform a
    # movement in reaction to a newly set velocity.
    ACTION_DELAY: ClassVar[float] = 0.05 #TODO

    def __init__(
        self,
        hostname: str = "pact-one",
        hilscher_port: int = 5555,
        command_port: int = None,
        speed_values: Tuple[int] = (
            # 100,
            # 400,
            # 800,
            # 1000,
            # 500,
            # 200,
            # 20,
            # 600,
            1 # should be ignored
        ),
        plot: bool = False,
        reposition_on_start: bool = True,
        block_swinging: bool = False,
        angle_terminals: bool = False,
        continuous: bool = True,
        backward_compatible: bool = False,
        sway_start:bool=True,
        evaluate:bool=False,
        cost_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        controller: Optional[Controller] = None,
    ):
        super().__init__(cost_function=cost_function)
        self.comms_initialized = False
        # self.ctx = zmq.Context()
        self.hostname = hostname
        # if not self.setup_config_subscription(command_port):
        #     LOG.warning("Subscribing to cmd socket failed; stop script manually.")
        self.comms = HardwareComms(hostname, hilscher_port)

        # Plant behavior
        self.continuous = continuous
        self.use_faulty_move = False  # use two cycles for a move
        if continuous and backward_compatible:
            LOG.warning(
                "Running in backwards compatible mode! Do not train new models in this mode!"
            )
            self.use_faulty_move = True
        action_name = self.action_type.name
        self.speeds = [0]
        if continuous and action_name != SwingupContinuousDiscreteAction.name:
            raise ValueError("Can't run in continuous mode without continuous action!")
        if not self.continuous:
            self.assert_valid_speeds(speed_values)
            self.speeds = speed_values
        self.reposition_on_start = reposition_on_start
        self._calibrated = False
        self._prev_ts: Optional[int] = None
        self.swing_block = block_swinging
        self.sway_start = sway_start
        self.zero_position = self.LEFT_SIDE
        # self.goal_position = 0
        self.eval = evaluate
        self.counter = 0
        # Bool to tell the plant to activate the angle terminal state
        self.off_the_stops = False
        self._hit_terminal = False  # flag to force recalibration if necessary

        # Task specific
        self.angle_terminals: bool = angle_terminals

        # Analysis
        self.plot = plot
        self._solved = False
        self.df_history = pd.DataFrame(columns=SwingupState.channels())

        # ZMQ returns
        #self._costfunc = cost_func
        self._controller = controller
        self._current_state = None

    @property
    def execution_registry(self) -> List[int]:
        return [
            # Commands.EXEC_SPEED5,  # left #3
            ## C/profinet-branch: Commands.EXEC_SPEED3,  # left #2
            Commands.EXEC_SPEED0,  # left #1
            Commands.EXEC_SPEED1,  # stop
            Commands.EXEC_SPEED2,  # right #1
            ## C/profinet-branch: Commands.EXEC_SPEED4,  # right #2
            # Commands.EXEC_SPEED6,  # right #3
        ]

    @staticmethod
    def assert_valid_speeds(speeds) -> None:
        """Assert all provided speeds are positive and nonzero."""
        assert all(s > 0 for s in speeds), "Read the plant docstring!"

    def setup_config_subscription(self, port: int) -> bool:
        """Subscribe to the command topic from the PACT frontend.

        Returns:
            True if successful.
        """
        try:
            self.config_socket = self.ctx.socket(zmq.SUB)
            self.config_socket.connect(f"tcp://{self.hostname}:{port}")
            self.config_socket.subscribe("cmd")
        except zmq.ZMQError:
            return False
        return True

    def initialize_comms(self) -> None:
        """Start zmq comms and set up initial parameters."""
        self.comms.send(Commands.RESET_ALL)
        self.comms.receive()
        self.comms.send(Commands.HERTZ, self.ACTION_DELAY * 1000)
        self.comms.receive()
        self.set_speeds(self.speeds)

    def assert_continue_run(self) -> None:
        """Check for kill message from Config UI.

        Kills are controlled by raising a KeyboardInterrupt. This is caught by
        the Cycle Manager and closes the loop down gracefully.
        """
        # try: #TODO: Disalbe this if setup config sub failed?
        #     topic, msg = self.config_socket.recv_multipart(flags=zmq.NOBLOCK)
        #     if topic == b"cmd":
        #         try:
        #             if int(msg) == Commands.SHUTDOWN:
        #                 raise KeyboardInterrupt
        #         except ValueError:  # msg is not int-coercible
        #             pass
        # except zmq.error.Again:
        #     pass
        pass

    @classmethod
    def calculate_cycle_time(cls, n_obs: int) -> float:
        """Calculate real plant time for n number of observations."""
        return n_obs * cls.ACTION_DELAY

    def convert_angle(self, angle: int) -> Tuple[float, float, int]:
        """Convert angles from PLC units to sine, cosine, and angle.

        The PLC sends angles in a range between [0,600]. This function
        converts those angles to and from the [-pi, pi] range, with 0
        being up vertically.
        """
        angle = angle - ((angle // self.TICKS_PER_CIRCLE) * self.TICKS_PER_CIRCLE)
        angle = np.pi / (self.TICKS_PER_CIRCLE / 2) * angle - np.pi
        return np.sin(angle), np.cos(angle), angle

    def read_state(self) -> Tuple[int, float, float, float, int]:
        """Read the current state from the plant.

        Angles are in a range [0, 600].  These are converted to radians
        with 0 being vertical, and decomposed into sine and cosine.

        We receive the left trigger stop in the state, but since the C code
        deals with calibration now, we do not need to know this anymore. It
        is therefore ignored. (state[2])

        Returns:
            state as: (position, sin(angle), cosine(angle), timestep, angle)
        """
        ts = time.time()  # basically instantaneous
        state = self.comms.receive()
        # self.comms.send_NOOP()
        # state = self.comms.receive() # double interaction time to .04 and get recent state
        state = state.split(" ")
        x = int(state[0])
        theta = int(state[1])
        sin, cos, theta = self.convert_angle(theta)

        x -= self.zero_position

        return x, sin, cos, ts, theta

    def lose_control(self) -> None:
        """Tell the proxy the plant will not control the cycle."""
        self.comms.send(Commands.RELINQUISH_CONTROL)

    def reset(self) -> None:
        """Reset the PLC's values and clear calibration."""
        self.assert_continue_run()
        self.comms.send(Commands.RESET_ALL)
        self.comms.receive()
        self.set_speeds(self.speeds)
        self._calibrated = False

    def turn_off_all_speeds(self) -> None:
        """Set all speed executions to 0 PLC-side."""
        self.comms.send(Commands.RESET_EXECUTES)
        self.comms.receive()

    def set_speeds(self, speeds: List[int]) -> None:
        """Set the velocities for all movement commands."""
        for i, speed in enumerate(speeds):
            self.comms.send(Commands.SET_SPEED_COMMANDS()[i], speed)
            self.comms.receive()

    def _get_extra_info(self) -> Tuple[Union[float, str], Union[float, str]]:
        """Returns cost and q for the current state."""
        # this is wrong / used wrong; when called in record history, current_state has not been set

        cost = ""
        q = ""
        if self._cost_function and self._current_state:
            cost = self._cost_function(self._current_state.as_array()[None, ...])[0]
        if self._controller:
            try:
                stack = self._controller._memory.stack[None, ...]
                stack = self._controller.normalizer.transform(stack)
                q = self._controller.min_model.predict(stack)

                q = q.ravel()[0]  # unpack
            except Exception as e:
                LOG.warning(f"Generating Q value failed.\n\t(Error: {e})")
        return cost, q

    def move(self, speed: int) -> None:
        """Use given speed to move (continuous actions)."""
        cost, q = self._get_extra_info()
        self.comms.send(Commands.CONT_SPEED, speed, cost, q)

    def faulty_move(self, speed: int) -> None:
        """Set the speed and execution separately in Python.

        Warning! This is faulty because it uses two cycles to execute
        a movement. Thus, the cycle time is doubled! This is only
        for backwards compatibility with older, working models, and
        when a working model is made with the correct continuous movement
        scheme, this functionality should be removed.
        """
        cost, q = self._get_extra_info()
        self.comms.send(Commands.SET_SPEED, speed, cost, q)
        self.comms.receive()
        self.comms.send(Commands.EXEC_SPEED2, speed, cost, q)

    def go_right(self) -> None:
        if self.continuous:
            raise PermissionError("Can't use specific movements in continuous mode")
        self.comms.send(Commands.EXEC_SPEED2)

    def go_left(self) -> None:
        if self.continuous:
            raise PermissionError("Can't use specific movements in continuous mode")
        self.comms.send(Commands.EXEC_SPEED0)

    def halt(self) -> None:
        """Stops the cart.

        In continuous mode, this speed is written as 0 so it can function properly."""
        self.comms.send(Commands.EXEC_SPEED1)

    def motor_on(self) -> None:
        """Turn the motor on."""
        self.assert_continue_run()
        self.comms.send(Commands.MOTOR_ON)
        self.comms.receive()
        LOG.info("Motor turned on.")

    def motor_off(self) -> None:
        """Turn the motor off"""
        self.assert_continue_run()
        self.comms.send(Commands.MOTOR_OFF)
        self.comms.receive()
        LOG.info("Motor turned off.")

    def calibrate(self) -> None:
        """Activate the homing calibration of the PLC."""
        self.assert_continue_run()
        LOG.info("Calibrating...")
        self.comms.send(Commands.CALIBRATE)
        self.comms.receive()
        LOG.info("Calibrated.")
        self._calibrated = True

    def return_to_center(self) -> None:
        """Returns the cart to the center of the axis."""
        self.assert_continue_run()
        LOG.info("Returning to center...")
        print("RETURNING TO CENTER")
        if not self.calibrated:
            LOG.warning("Trying to go to center of an uncalibrated axis!")
            raise RuntimeWarning
        self.comms.send(Commands.CENTER)
        self.comms.receive()
        LOG.info("Returned to center.")
        print("RETURNED TO CENTER")

        #TODO I added the dist stops, added reset params, and slowed down the operator to 150 from 250

    def check_initial_state(self, state: Optional[SwingupState]) -> SwingupState:
        self.assert_continue_run()
        assert self.episode_steps == 0
        self.comms.send_NOOP()  # need to send to receive
        x, sin, cos, ts, theta = self.read_state()
        time.sleep(self.ACTION_DELAY)
        self.comms.send_NOOP()
        x_, sin_, cos_, ts_, theta_ = self.read_state()
        x_dot = (x_ - x) #/ (ts_ - ts)
        theta_dot = angle_slope(theta_, theta) #/ (ts_ - ts)
        self._prev_ts = ts_
        right_dist = abs(self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET - x_ + self.zero_position)
        left_dist = abs(self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET - x_ + self.zero_position)
        #operator = 0.0 # 150 if x_ < 0 else -150  #TODO 250
        obs = OrderedDict(
            cart_position=x_,
            cart_velocity=x_dot,
            pole_sine=sin_,
            pole_cosine=cos_,
            pole_velocity=theta_dot,
            pole_angle=theta_,
#            true_position=x_+self.zero_position,
            dist_left = left_dist,
            dist_right = right_dist,
            #operator=operator,
            direction_ACT=0.0, # we executed a NOOP before read_state, thus 0.0 is correct direction leading to this state measurement
        )
        self._current_state = self.state_type(obs, meta=dict(ts=ts_))
        self._record_history(*obs.values())
        return self._current_state

    def _get_next_state(
        self, state: SwingupState, action: SwingupContinuousDiscreteAction
    ) -> SwingupState:
        self.assert_continue_run()

        self.counter += 1
        #if self.eval and self.counter % 250 == 0:
        #    self.counter = 0
        #    if self.zero_position < 3000:
        #        print("GOAL CHANGES TO RIGHT")
        #        self.zero_position = self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET - 400
        #    else:
        #        print("GOAL CHANGES TO LEFT")
        #        self.zero_position = self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET + 400

        assert state == self._current_state
        direction = action["direction"]

        ##### MIMIC OPERATOR
        # print(state["cart_position"])  #TODO 250
        #if state["cart_position"] < -300:
        #    operator = 150
        #elif state["cart_position"] > 300:
        #    operator = -150
        #else:
        #    operator = 0
            # operator = int(state["cart_position"] / 2)
            # if state["cart_position"] < 50:
            #     operator = 0

        #direction = (.5 * operator) + (.5 * direction)
        #####

        with CM["get-state/vel"]:
            if self.continuous:
                if self.use_faulty_move:
                    # Use backward compatible speed
                    self.faulty_move(direction)
                else:
                    # Set the speed and execute C side
                    self.move(direction)
            else:
                # The mapping is defined in the execution_registry;
                # it is not known what is left or right at the outset
                self.comms.send(self.execution_registry[int(direction)])

        # Get previous measurements to compute slopes on later.
        x = float(self._current_state["cart_position"])
        theta = cast(int, self._current_state["pole_angle"])

        with CM["get-state/readstate"]:
            # SL: I doubt this handling of the overflow is correct / useful--> should probably
            #     go to comms before handling any offsets (zero position) and should either clip
            #     to zero or (better from Markov / RL standpoint) allow negative positions
            #     and also allow positive positions outside range (state space doesnt end there,
            #     positions outside x-work in x-minus are possible)
            #
            # Build new state given new readings.
            x_, sin_, cos_, ts_, theta_ = self.read_state()
            true_x = x_+self.zero_position  # SL: read encoder position is moved by -zero_position, move it back here for the bounds check
            # if x_ > 10000:
            #     LOG.warning("Overflow in position!")
            #     x_ = 0  # prevent overflow
            if true_x > 20000: # SL: larger value, to be sure (zero shift).
                LOG.warning("Underflow in position!")
                true_x = 0  # prevent overflow
                x_ = - self.zero_position # SL, did this, quick and ugly fix for the moment, but should better become negative number, in order to give a correct (markov!) velocity estimate

            x_dot = (x_ - x) #/ (ts_ - self._prev_ts)
            theta_dot = angle_slope(theta_, theta) #/ (ts_ - self._prev_ts)
            self._prev_ts = ts_
            right_dist = self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET -true_x
            left_dist = - (self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET - true_x)
            # print("L:\t", left_dist, "\tP:", true_x, "\tR:", right_dist)
            obs = dict(
                cart_position=x_,
                cart_velocity=x_dot,
                pole_sine=sin_,
                pole_cosine=cos_,
                pole_velocity=theta_dot,
                pole_angle=theta_,
#                true_position=true_x,
                dist_left = left_dist,
                dist_right=right_dist,
                #operator=operator,
                direction_ACT=direction,
            )

        # Determine if terminal state
        left = self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET #- self.zero_position
        right = self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET# - self.zero_position
        bad_position = not (left <= true_x <= right)
        overspeed = np.abs(theta_dot) > 20#0.65
        bad_angle = False # SL: clarify, why this was // cos_ > 0#-.707# .82
        if self.angle_terminals:
            if np.abs(theta_) < 0.1 and not self.off_the_stops:
                self.off_the_stops = True
                LOG.debug("Activating angle terminal state...")
            bad_angle = False
            if self.off_the_stops:  # -.68067, .70162
                bad_angle = np.abs(theta_) >= 0.67
            terminal = bad_angle or bad_position
        else:
            terminal = bad_position # SL: or bad angles must have been wrong here //or bad_angle #or overspeed
            if bad_position:
                print("\t\t\tTERMINAL DUE TO")
                print("\t\t\tCART_POSITION OUT OF BOUNDS")
            # if overspeed:
            #     print("\t\t\tTERMINAL DUE TO")
            #     print("\t\t\tOVERSPEED")
            # if bad_angle:
            #    print("\t\tTERMINAL DUE TO")
            #    print("\t\tBAD ANGLE")

        # Stop the cart if in terminal state
        if terminal:
            self.halt()
            LOG.info(
                f"Stopped the cart due to termination "
                f"({round(x_, 2)}, {round(theta_, 2)})."
            )
            # Only bad position terminals need resetting
            self._hit_terminal = True if bad_position else False

        next_state = self.state_type(obs, terminal=terminal, meta=dict(ts=ts_))
        self._current_state = next_state  # this should not be necessary, because base plant does it
        self._record_history(*obs.values()) # must be after setting current state (above), because it uses self.current_state (unfortunately, internally, for getting costs). 
             # SL: recording history this way has the problem, that we display the PREVIOUS action aligned with the present state, NOT the action selected by the agent in repsonse to this state. This will lead to misinterpreatations of the graph and also lead to underestimating the delay by one cylce.
        return self._current_state

    def enforce_pole_down(self, consecutive_steps: int = 5) -> None:
        """Blocks until pole stops moving and hangs downwards.

        Args:
            consecutive_steps: How many 1/10s of a second to have the same pole
                               angle before saying it is stopped.
        """
        self.assert_continue_run()
        if self.angle_terminals:
            LOG.warning("Can not enforce downwards pole with the balancing task.")
            return
        stopped = 0
        self.comms.send_NOOP()
        _, _, cos, _, _ = self.read_state()
        LOG.info("Waiting until pole is stable...")
        while stopped < consecutive_steps:  # Requires int angles
            time.sleep(0.1)
            self.comms.send_NOOP()
            _, _, cos_, _, _ = self.read_state()
            if abs(cos_ - cos) < 0.1 and cos_ < 0 and cos < 0:
                stopped += 1
            else:
                # Reset angle and counter
                cos = cos_
                stopped = 0
        LOG.info("Pole stabilized.")

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def is_solved(self) -> bool:
        return self._solved

    def _record_history(
        self,
        x_: float,
        x_dot: float,
        sine_: float,
        cosine_: float,
        theta_dot: float,
        theta: float,
        ld,
        rd,
        #op,
#        x_true: float,
        vel: float,
    ) -> None:
        cost, q = self._get_extra_info()

        """Record a state in the local history for analysis"""
        self.df_history.loc[self.episode_steps, "cart_position"] = x_
        self.df_history.loc[self.episode_steps, "cart_velocity"] = x_dot
        self.df_history.loc[self.episode_steps, "pole_sine"] = sine_
        self.df_history.loc[self.episode_steps, "pole_cosine"] = cosine_
        self.df_history.loc[self.episode_steps, "pole_velocity"] = theta_dot
        self.df_history.loc[self.episode_steps, "pole_angle"] = theta
        #self.df_history.loc[self.episode_steps, "true_position"] = x_true
        self.df_history.loc[self.episode_steps, "left_dist"] = ld
        self.df_history.loc[self.episode_steps, "right_dist"] = rd
        #self.df_history.loc[self.episode_steps, "operator"] = op
        self.df_history.loc[self.episode_steps, "direction_ACT"] = vel
        self.df_history.loc[self.episode_steps, "cost"] = cost
        self.df_history.loc[self.episode_steps, "q"] = q

    def solve_condition(self, state: SwingupState) -> bool:
        """Return True if potentially solved.

        If angle < .15 rad (~8 deg) off vertical at end of episode,
        the task is potentially solved.  It may have just ended within
        the goal region, and so the saved model needs to be double checked.
        """
        angle_goal = 0.15

        if np.abs(state["pole_angle"]) < angle_goal:
            return True
        return False

    def get_ready(self) -> None:
        """Calibrate (if necessary) and bring the cart to the center of the axis."""
        # Turn the motor on. The extra timing here is to ensure return to center starts.
        time.sleep(0.05)
        self.motor_on()
        time.sleep(0.05)
        # Turn off all speeds or else return to center won't work without a reset
        self.turn_off_all_speeds()
        # Only calibrate if we aren't already calibrated, to save time
        if not self.calibrated:
            self.calibrate()
        self.return_to_center()
        #self.comms.receive()
        # Check if returning to center actually worked and if not, recalibrate
        self.comms.send_NOOP()
        position = self.read_state()[0]
        #if not (self.CENTER - 400 - self.zero_position < position < self.CENTER + 400- self.zero_position):
        #    print("\tRecentering...")
        #    LOG.warning("Centering failed, recalibrating...")
        #    self._calibrated = False
        #    self.get_ready()

    def move_offcenter(self):
        left = self.zero_position < self.CENTER
        print("GOAL POSITION IS", f"{'left' if left else 'right'}")
        if not left:
            print("Moving left...")
            self.move(-100)
            time.sleep(2)
            self.read_state()
            self.move(0)
            print("Pos", self.read_state()[0])
            self.comms.send_NOOP()
            s = self.read_state()
            print("Pos", s[0])
        else:
            print("Moving right...")
            self.move(100)
            time.sleep(2)
            self.read_state()
            self.move(0)
            print("Pos", self.read_state()[0])
            self.comms.send_NOOP()
            s = self.read_state()
            print("Pos", s[0])
        if np.random.random() < .33 or self.eval:
            print("JIGGLING")
            for _ in range(1):
                self.move(-200)
                time.sleep(.32)
                self.read_state()
                self.move(200)
                time.sleep(.32 * 2)
                self.read_state()
                self.move(-200)
                time.sleep(.32)
                self.read_state()
            self.move(0)
            print("Pos", self.read_state()[0])
            self.comms.send_NOOP()
            print("Pos", self.read_state()[0])
        print("DONE WITH SETUP")
        return

    def notify_episode_starts(self) -> bool:
        self.assert_continue_run()
        super().notify_episode_starts()
        self.df_history = pd.DataFrame(columns=SwingupState.channels()) # SL: reset data for plotting

        self.comms.notify_episode_starts()
        self.initialize_comms()
        self.get_ready()
        if self.sway_start:
            # Goal position is tested for > .5 to determine
            # left or right goals; it stays static throughout
            # the entire episode.
            # self.goal_position = np.random.random()
            # Convert positions to move the goal region around
            if np.random.random() < .5:  # goal on the left
                self.zero_position = self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET + 400
            else:  # goal on the right
                self.zero_position = self.RIGHT_SIDE - self.TERMINAL_RIGHT_OFFSET - 400
            if self.eval:
                self.zero_position = self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET + 400
            else:
                pass
                # self.zero_position = self.LEFT_SIDE + self.TERMINAL_LEFT_OFFSET + 400
            print("ZERO POSITION IS", self.zero_position)
            self.move_offcenter()
        if self.swing_block:
            self.enforce_pole_down()
        # For holding the pole up...
        # for i in list(range(0,4)[::-1]):
        #     print(f"{i}\t{i}\t{i}\t{i}\t{i}\t{i}\t{i}")
        #     time.sleep(1)
        return True

    def notify_episode_stops(self) -> bool:
        #print(f"####### last cost in df_history { self.df_history['cost'].iloc[-1] }")
    
        #self.df_history["left"] = self.df_history["cart_position"] <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET
        #print(self.df_history)

        self.assert_continue_run()
        time.sleep(0.1)
        try:
            self.comms.receive()
        except zmq.error.ZMQError:
            pass
        self.halt()
        self.comms.receive()
        if not self.reposition_on_start or self._hit_terminal:
            LOG.info("Recalibration requested at end of episode.")
            self.reset()  # erase calibration
        self.motor_off()
        self.lose_control()
        self.comms.notify_episode_stops()
        # Comms are disconnected after this point
        self._hit_terminal = False
        self.off_the_stops = False
        self._solved = self.solve_condition(self._current_state)

        if self.plot:
            plot_swingup_state_history(self)
        return True

    def __del__(self):
        # If still connected to the plant, stop and disconnect
        if self.comms.active:
            self.halt()
            self.comms.receive()
            self.motor_off()
            self.comms.notify_episode_stops()


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
    cost = None
    if sart_path:
        with SARTReader(sart_path) as sart:
            sart = sart.load_full_episode()
            x = sart[0][:, 0]
            x_s = sart[0][:, 1]
            t = sart[0][:, 2]
            td = sart[0][:, 3]
            a = sart[0][:, 4]
    else:
        plant = cast(SwingupPlant, plant)
        x = plant.df_history.cart_position
        x_s = plant.df_history.cart_velocity
        t = plant.df_history.pole_angle
        td = plant.df_history.pole_velocity
        a = plant.df_history.direction_ACT
        cost = plant.df_history.cost

    fig, axs = plt.subplots(5, figsize=(10, 8))
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

    axs[2].plot(x, label="pole_velocity")
    axs[2].set_title("pole_velocity")
    axs[2].set_ylabel("Angular Vel")
    axs[2].legend()

    axs[3].plot(a, label="Action")
    axs[3].axhline(0, color="grey", linestyle=":")
    axs[3].set_title("Control")
    axs[3].set_ylabel("Velocity")
    axs[3].legend(loc="upper left")
    axs2b = axs[3].twinx()
    axs2b.plot(x_s, color="black", alpha=0.4, label="True Velocity")
    axs2b.set_ylabel("Steps/s")
    axs2b.legend(loc="upper right")

    if cost is not None:
        axs[4].plot(plant.df_history.cost, label="cost")
        axs[4].set_title("cost")
        axs[4].set_ylabel("cost")
        axs[4].legend()

    plt.suptitle("NFQ Controller on Physical Swingup Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # sart_path = (
    #     "../../../../examples/rl/hamburg_cartpole/data/"
    #     "test-sart.h5"
    # )
    # plot_swingup_state_history(None, sart_path=sart_path)
    # exit()
    ## C/profinet-branch: VELOCITIES = [20, 100, 600]
    VELOCITIES = [200]
    plant = SwingupPlant(
        "pact-one", 5555, 5556, speed_values=VELOCITIES, continuous=True
    )
    plant.comms.notify_episode_starts()
    plant.reset()
    print("Calibrating...")
    plant.calibrate()
    print("Centering...")
    plant.return_to_center()
    print("Done.")

    def print_state():
        plant.motor_off()
        plant.comms.send_NOOP()
        ts_ = 0
        while True:
            x, sin, cos, ts, theta = plant.read_state()
            diff = ts - ts_
            print(f"X: {x}\tSIN: {round(sin,3)}\tCOS: {round(cos,3)}\tTHETA: {round(theta,3)}\tTS: {ts} (delta: {round(diff,4)})")
            time.sleep(0.2)
            ts_ = ts
            plant.comms.send_NOOP()

    def test_continuous_movement():
        """Test continuous speeds C side."""
        plant.continuous = True
        print("Testing continuous movement C side.")
        print("Moving right 10")
        plant.move(10)
        plant.comms.receive()
        time.sleep(1)
        print("Stopping...")
        plant.move(0)
        plant.comms.receive()
        time.sleep(1)
        print("Moving left 10")
        plant.move(-10)
        plant.comms.receive()
        time.sleep(1)
        print("Stopping...")
        plant.move(0)
        plant.comms.receive()
        time.sleep(1)

        print("Testing cycle times...")
        start = time.time()
        plant.move(50)
        while time.time() - start < 2:
            state = plant.read_state()
            print(f"T: {time.time()}  |  X: {state[0]}  ")
            plant.comms.send_NOOP()
        start = time.time()
        plant.comms.receive()
        plant.move(-50)
        while time.time() - start < 2:
            state = plant.read_state()
            print(f"T: {time.time()}  |  X: {state[0]}  ")
            plant.comms.send_NOOP()

        plant.comms.receive()
        plant.move(0)
        plant.comms.receive()
        print("Complete.")

    def test_discrete_movement():
        """Move with all three speed blocks."""
        # 0 Commands.EXEC_SPEED5,  # left #3
        # 1 Commands.EXEC_SPEED3,  # left #2
        # 2 Commands.EXEC_SPEED0,  # left #1
        # 3 Commands.EXEC_SPEED1,  # stop
        # 4 Commands.EXEC_SPEED2,  # right #1
        # 5 Commands.EXEC_SPEED4,  # right #2
        # 6 Commands.EXEC_SPEED6,  # right #3
        print("Testing movement.")
        print("Moving right 1")
        # plant.comms.send(plant.execution_registry[4])
        plant.comms.send(plant.execution_registry[2])
        plant.comms.receive()
        time.sleep(1)
        print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        plant.comms.send(plant.execution_registry[1])
        plant.comms.receive()
        time.sleep(1)

        print("Moving left 1")
        # plant.comms.send(plant.execution_registry[2])
        plant.comms.send(plant.execution_registry[0])
        plant.comms.receive()
        time.sleep(1)
        print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        plant.comms.send(plant.execution_registry[1])
        plant.comms.receive()
        time.sleep(1)

        # print("Moving right 2")
        # plant.comms.send(plant.execution_registry[5])
        # plant.comms.receive()
        # time.sleep(0.5)
        # print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        # plant.comms.receive()
        # time.sleep(1)

        # print("Moving left 2")
        # plant.comms.send(plant.execution_registry[1])
        # plant.comms.receive()
        # time.sleep(0.5)
        # print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        # plant.comms.receive()
        # time.sleep(1)

        # print("Moving right 3")
        # print("DOES NOT WORK")  # TODO
        # plant.comms.send(plant.execution_registry[6])
        # plant.comms.receive()
        # time.sleep(0.5)
        # print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        # plant.comms.receive()
        # time.sleep(1)

        # print("Moving left 3")
        # plant.comms.send(plant.execution_registry[0])
        # plant.comms.receive()
        # time.sleep(0.5)
        # print("Stopping")
        # plant.comms.send(plant.execution_registry[3])
        # plant.comms.receive()
        # time.sleep(1)

    def move_back_and_forth():
        print("Testing back and forth angles.")
        for i in range(20):
            print("Moving right")
            plant.move(200)
            state = plant.read_state()  # x, sin, cos, ts, theta
            print("sin", state[1], "cos", state[2], "theta", state[-1])
            time.sleep(1)
            print("Stopping...")
            plant.move(0)
            state = plant.read_state()  # x, sin, cos, ts, theta
            print("sin", state[1], "cos", state[2], "theta", state[-1])
            time.sleep(1)
            print("Moving left 10")
            plant.move(-200)
            state = plant.read_state()  # x, sin, cos, ts, theta
            print("sin", state[1], "cos", state[2], "theta", state[-1])
            time.sleep(1)
            print("Stopping...")
            plant.move(0)
            state = plant.read_state()  # x, sin, cos, ts, theta
            print("sin", state[1], "cos", state[2], "theta", state[-1])
            time.sleep(1)

    def recenter_check():
        print("Checking ability to recenter")
        for i in range(20):
            print("Moving right")
            plant.move(200)
            state = plant.read_state()  # x, sin, cos, ts, theta
            time.sleep(1)
            print("Stopping...")
            plant.move(0)
            print("RECENTERING")
            state = plant.read_state()  # x, sin, cos, ts, theta
            plant.return_to_center()
            time.sleep(1)
            print("Moving left")
            plant.move(-200)
            state = plant.read_state()  # x, sin, cos, ts, theta
            time.sleep(2)
            print("Stopping...")
            plant.move(0)
            state = plant.read_state()  # x, sin, cos, ts, theta
            time.sleep(1)
            print("RECENTERING")
            plant.return_to_center()

    def move_offcenter_check():
        print("Checking offcenter move logic")
        if np.random.random() < .5:
            print("Moving left...")
            # plant.move(-100)
            # time.sleep(2)
            # plant.read_state()
            # plant.move(0)
            # print("Pos", plant.read_state()[0])
            plant.comms.send_NOOP()
            s = plant.read_state()
            print("Pos", s[0])

            if s[4] < .075:
                print("JIGGLING")
                for _ in range(3):
                    plant.move(500)
                    time.sleep(.08)
                    plant.read_state()
                    plant.move(-500)
                    time.sleep(.16)
                    print("Pos", plant.read_state()[0])
                plant.move(0)
                plant.read_state()
                plant.comms.send_NOOP()
                print("Pos", plant.read_state()[0])
        else:
            print("Moving right...")
            # plant.move(100)
            # time.sleep(2)
            # plant.read_state()
            # plant.move(0)
            # print("Pos", plant.read_state()[0])
            plant.comms.send_NOOP()
            s = plant.read_state()
            print("Pos", s[0])

            if s[4] < .075:
                print("JIGGLING")
                for _ in range(3):
                    plant.move(-500)
                    time.sleep(.08)
                    plant.read_state()
                    plant.move(500)
                    time.sleep(.16)
                    print("Pos", plant.read_state()[0])
                plant.move(0)
                plant.read_state()
                plant.comms.send_NOOP()
                print("Pos", plant.read_state()[0])


    # recenter_check()
    # test_discrete_movement()
    # test_continuous_movement()
    # move_back_and_forth()
    for i in range(4):
        move_offcenter_check()
        plant.reset()
        print("Calibrating...")
        plant.calibrate()
        print("Centering...")
        plant.return_to_center()
        print("Done.")
    print_state()
    plant.motor_off()
