from typing import Literal, Optional

import numpy as np
import zmq
from psipy.rl.core.plant import Action, Plant, State

"""
    ZMQ Proxy Plant for Autocrane

    This module defines the AutocraneZMQProxyPlant class, which interfaces with the autocrane core via ZeroMQ. It includes state and action definitions specific to the autocrane, facilitating remote control and monitoring of the crane's components (gantry, hoist, trolley, and grapple) through a discrete action space.

    This plant is ready to be used with the psipy RL framework and can be run in a control loop as any other plant.

    On the autocrane core side, the ZMQProxy - Behavior needs to be activated.
    It serves as the endpoint for sending states and receiving actions. The proxy expects an action to be received within the same cycle the state was sent. It will ignore and drop any actions received late, or containing a different cycle number. Presently, a cycle is 50ms max on the core side, with even less time for the ZMQProxy to be able to wait. It's important to note that the core will not wait for the proxy to be ready or reply; it will just continue its control cycles and execution, sending out new state information every cycle, independently of whether the proxy read this information and did or did not reply in time.

    Some specification of the minicrane, for the trolley axis:

    speed_max_possible: 0.2680195074658086
    speed_max_allowed: 0.2680195074658086
    speed_non_zero_min: 0.02864420251680448
"""


class AutocraneState(State):
    _channels = (  # SI units, positions are in meters, velocities are in meters per second
        "active",
        "gantry_pos",
        "hoist_pos",
        "trolley_pos",
        "grapple_pos",
        "gantry_vel",
        "hoist_vel",
        "trolley_vel",
        "grapple_vel",
        "grapple_sway_trolley",
        "grapple_sway_trolley_vel",
        "grapple_sway_gantry",
        "grapple_sway_gantry_vel",
        "grapple_loaded",
        "gantry_limit_dist_left",
        "gantry_limit_dist_right",
        "trolley_limit_dist_left",
        "trolley_limit_dist_right",
        "hoist_limit_dist_up",
        "hoist_limit_dist_down",
        "gantry_set_point_delta",
        "trolley_set_point_delta",
        "hoist_set_point_delta",
        "gantry_target_vel_ACT",
        "hoist_target_vel_ACT",
        "trolley_target_vel_ACT",
        "grapple_target_vel_ACT",
        "weight",
    )


class AutocraneAction(Action):
    dtype = "discrete"


class AutocraneTrolleyAction(AutocraneAction):
    dtype = "discrete"
    channels = ("trolley_target_vel",)
    legal_values = ((-0.268, 0.0, 0.268),)


class ExtendedAutocraneTrolleyAction(AutocraneAction):
    dtype = "discrete"
    channels = ("trolley_target_vel",)
    legal_values = ((-0.268, -0.082, 0.0, 0.082, 0.268),)


class AutocraneGantryAction(AutocraneAction):
    dtype = "discrete"
    channels = ("gantry_target_vel",)
    legal_values = ((-1.0, 0.0, 1.0),)


class ExtendedAutocraneGantryAction(AutocraneAction):
    dtype = "discrete"
    channels = ("gantry_target_vel",)
    legal_values = ((-1.0, -0.5, 0.0, 0.5, 1.0),)


class AutocraneTrolleyHoistAction(AutocraneAction):
    dtype = "discrete"
    channels = (
        "trolley_target_vel",
        "hoist_target_vel",
    )
    legal_values = (
        (-0.268, -0.082, 0.0, 0.082, 0.268),
        (-0.095, 0.0, 0.095),
    )


class AutocraneDiscreteAction(AutocraneAction):
    dtype = "discrete"
    channels = (
        "gantry_target_vel",
        "trolley_target_vel",
        "hoist_target_vel",
    )
    legal_values = (  # speed is in meters per second
        (-1.0, 0.0, 1.0),  # max vel is 1.00 m/s
        (-0.268, 0.0, 0.268),
        (-0.095, 0.0, 0.095),
    )


class AutocraneZMQProxyPlant(Plant[AutocraneState, AutocraneAction]):
    """
    A Plant class that interfaces with an Autocrane system via ZMQ.

    This class acts as a proxy between the reinforcement learning environment
    and the actual Autocrane system. It communicates with the Autocrane using
    ZeroMQ (ZMQ) for sending actions and receiving states.

    Attributes:
        state_type (Type[AutocraneState]): The type of state used by this plant.
        action_type (Type[AutocraneDiscreteAction]): The type of action used by this plant.
        context (zmq.Context): The ZMQ context for creating sockets.
        state_socket (zmq.Socket): The ZMQ subscriber socket for receiving states.
        action_socket (zmq.Socket): The ZMQ publisher socket for sending actions.
        _last_cycle_number (int): Keeps track of the last cycle number received.
        _randomize_set_points (bool): Whether to randomize set points.
        trolley_min (float): The minimum position of the trolley.
        trolley_max (float): The maximum position of the trolley.

    Properties:
        set_point_trolley (float): The current set point for the trolley position.

    Methods:
        set_random_set_point(): Draws a random set point for the trolley position from the allowed range.
        move_away_from_limits(): Moves the trolley away from the limits.

    The plant communicates with the Autocrane system using two ZMQ sockets:
    1. A subscriber socket for receiving state information.
    2. A publisher socket for sending action commands.

    It also manages set points for the trolley, which can be randomized if desired.
    """

    state_type = AutocraneState
    action_type = AutocraneAction

    def __init__(
        self,
        state_sub_address: str = "tcp://192.168.50.25:7555",
        action_pub_address: str = "tcp://192.168.50.25:7556",
        randomize_set_points: bool = True,
        alternating_set_points: bool = False,
        hoist_active: bool = False,
        action_type: Optional[AutocraneAction] = None,
        axis: Literal["trolley", "gantry"] = "trolley",
        sway_terminal_margin: float = 0.3,
        good_terminal_sway_margin: float = 0.02,
        good_terminal_distance_margin: float = 0.05,
        good_terminal_consecutive_steps: int = 10,
        **kwargs,
    ):
        """
        Initializes the AutocraneZMQProxyPlant.

        Args:
            state_sub_address (str): The address of the ZMQ subscriber socket for receiving states.
            action_pub_address (str): The address of the ZMQ publisher socket for sending actions.
            randomize_set_points (bool): Whether to randomize set points.
            axis (str): Which axis to control: "trolley" or "gantry" (default: "trolley").
            sway_terminal_margin (float): Sway threshold for bad terminal state (default: 0.3).
            good_terminal_sway_margin (float): Maximum sway angle for good terminal state (default: 0.02).
            good_terminal_distance_margin (float): Maximum distance to setpoint for good terminal state (default: 0.05).
            good_terminal_consecutive_steps (int): Number of consecutive steps within margins to trigger good terminal (default: 10).
        """
        self.hoist_active = hoist_active
        self._axis = axis

        super().__init__(**kwargs)
        self.context = zmq.Context()

        # Subscriber socket for receiving states
        self.state_socket = self.context.socket(zmq.SUB)
        self.state_socket.connect(state_sub_address)
        self.state_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Publisher socket for sending actions
        self.action_socket = self.context.socket(zmq.PUB)
        self.action_socket.setsockopt(zmq.CONFLATE, 1)

        self.action_socket.connect(action_pub_address)

        self._last_cycle_number = 0

        # Set point related attributes
        self._randomize_set_points = randomize_set_points
        self._alternating_set_points = (
            alternating_set_points and not randomize_set_points
        )
        self.next_option = 0
        self.trolley_min: float | None = None
        self.trolley_max: float | None = None
        self.hoist_min: float | None = None
        self.hoist_max: float | None = None

        self._current_state: AutocraneState | None = None

        self._gantry_set_point: float | None = None
        self._trolley_set_point: float | None = None
        self._hoist_set_point: float | None = None

        # Terminal state margins
        self._sway_terminal_margin = sway_terminal_margin
        self._good_terminal_sway_margin = good_terminal_sway_margin
        self._good_terminal_distance_margin = good_terminal_distance_margin
        self._good_terminal_consecutive_steps = good_terminal_consecutive_steps
        self._consecutive_good_steps = 0
        self._is_good_terminal = False

    @property
    def set_point_trolley(self):
        """
        The current set point for the trolley position.
        """
        return self._trolley_set_point

    @set_point_trolley.setter
    def set_point_trolley(self, position: float):
        """
        Sets the set point for the trolley position.
        """
        print(f"Setting set point to {position}")
        self._trolley_set_point = position

    @property
    def set_point_hoist(self):
        """
        The current set point for the hoist position.
        """
        return self._hoist_set_point

    @set_point_hoist.setter
    def set_point_hoist(self, position: float):
        """
        Sets the set point for the hoist position.
        """
        print(f"Setting set point to {position}")
        self._hoist_set_point = position

    @property
    def set_point_gantry(self):
        """
        The current set point for the gantry position.
        """
        return self._gantry_set_point

    @set_point_gantry.setter
    def set_point_gantry(self, position: float):
        """
        Sets the set point for the gantry position.
        """
        print(f"Setting gantry set point to {position}")
        self._gantry_set_point = position

    def set_alternating_set_point(self):
        """
        Sets the set point for the controlled axis (trolley or gantry) to the alternating set point.
        """
        if self._axis == "gantry":
            if self.gantry_min is None or self.gantry_max is None:
                return
            gantry_buffer = 0.6
            gantry_center = (self.gantry_min + self.gantry_max) / 2.0
            hoist_low = (self.hoist_min + 0.4) if self.hoist_min is not None else 0.0
            hoist_high = (self.hoist_max - 0.4) if self.hoist_max is not None else 0.0
            hoist_mid = (
                ((self.hoist_min + self.hoist_max) / 2.0)
                if (self.hoist_min is not None and self.hoist_max is not None)
                else 0.0
            )
            options = (
                (self.gantry_min + gantry_buffer, hoist_low),
                (self.gantry_max - gantry_buffer, hoist_low),
                (gantry_center, hoist_mid),
                (self.gantry_min + gantry_buffer, hoist_high),
                (self.gantry_max - gantry_buffer, hoist_high),
            )
            self.set_point_gantry = options[self.next_option][0]
            if (
                self.hoist_active
                and self.hoist_min is not None
                and self.hoist_max is not None
            ):
                self.set_point_hoist = options[self.next_option][1]
            self.next_option = (self.next_option + 1) % len(options)
            print(
                f"Setting alternating set points to gantry={self.set_point_gantry}, hoist={getattr(self, '_hoist_set_point', None)}"
            )
            return

        if (
            self.hoist_min is None
            or self.hoist_max is None
            or self.trolley_min is None
            or self.trolley_max is None
        ):
            return

        options = (
            (self.trolley_min + 0.6, self.hoist_min + 0.4),
            (self.trolley_max - 0.6, self.hoist_min + 0.4),
            (
                (self.trolley_min + self.trolley_max) / 2.0,
                (self.hoist_min + self.hoist_max) / 2.0,
            ),  # point in the center of the Z or X-shaped movement / cross of the X
            (self.trolley_min + 0.6, self.hoist_max - 0.4),
            (self.trolley_max - 0.6, self.hoist_max - 0.4),
        )
        self.set_point_trolley = options[self.next_option][0]

        if self.hoist_active:
            self.set_point_hoist = options[self.next_option][1]

        self.next_option = (self.next_option + 1) % len(options)

        print(
            f"Setting alternating set points to {self.set_point_trolley}, {self.set_point_hoist}"
        )

    def set_random_set_point(self):
        """
        Draws a random set point for the controlled axis (trolley or gantry) from the allowed
        range. Set points will be at least 20% of the movement range away from
        the limits and at least 1 meter away from the current position (for trolley/gantry).
        """
        if self._axis == "gantry":
            if self.gantry_min is None or self.gantry_max is None:
                return
            buffer = 0.2 * (self.gantry_max - self.gantry_min)
            valid_min = self.gantry_min + buffer
            valid_max = self.gantry_max - buffer
            pos_key = "gantry_pos"
            if self._current_state is not None and pos_key in self._current_state:
                current_pos = self._current_state[pos_key]
                min_distance = 1.0
                left_range_max = min(current_pos - min_distance, valid_max)
                right_range_min = max(current_pos + min_distance, valid_min)
                valid_ranges = []
                if valid_min < left_range_max:
                    valid_ranges.append((valid_min, left_range_max))
                if right_range_min < valid_max:
                    valid_ranges.append((right_range_min, valid_max))
                if valid_ranges:
                    range_min, range_max = valid_ranges[
                        np.random.randint(len(valid_ranges))
                    ]
                    self.set_point_gantry = np.random.uniform(range_min, range_max)
                else:
                    if current_pos - min_distance >= valid_min:
                        self.set_point_gantry = current_pos - min_distance
                    elif current_pos + min_distance <= valid_max:
                        self.set_point_gantry = current_pos + min_distance
                    else:
                        self.set_point_gantry = (valid_min + valid_max) / 2.0
            else:
                self.set_point_gantry = np.random.uniform(valid_min, valid_max)
            if self.hoist_min is not None and self.hoist_max is not None:
                self.set_point_hoist = np.random.uniform(
                    self.hoist_min + 0.1 * (self.hoist_max - self.hoist_min),
                    self.hoist_max - 0.1 * (self.hoist_max - self.hoist_min),
                )
            return

        if self.trolley_min is None or self.trolley_max is None:
            return

        # Calculate valid range with buffer from limits
        buffer = 0.2 * (self.trolley_max - self.trolley_min)
        valid_min = self.trolley_min + buffer
        valid_max = self.trolley_max - buffer

        # Make sure the set point is not too close to the limits and also
        # not too close to the current position (minimum distance of 1 meter)
        if self._current_state is not None and "trolley_pos" in self._current_state:
            current_trolley_pos = self._current_state["trolley_pos"]
            min_distance = 1.0

            # Calculate ranges that are at least min_distance away from current position
            left_range_min = valid_min
            left_range_max = min(current_trolley_pos - min_distance, valid_max)
            right_range_min = max(current_trolley_pos + min_distance, valid_min)
            right_range_max = valid_max

            # Collect valid ranges
            valid_ranges = []
            if left_range_min < left_range_max:
                valid_ranges.append((left_range_min, left_range_max))
            if right_range_min < right_range_max:
                valid_ranges.append((right_range_min, right_range_max))

            if valid_ranges:
                # Pick a random range and then a random point within it
                range_min, range_max = valid_ranges[
                    np.random.randint(len(valid_ranges))
                ]
                self.set_point_trolley = np.random.uniform(range_min, range_max)
            else:
                # Edge case: current position is too close to limits
                # Pick the farthest valid position
                if current_trolley_pos - min_distance >= valid_min:
                    self.set_point_trolley = current_trolley_pos - min_distance
                elif current_trolley_pos + min_distance <= valid_max:
                    self.set_point_trolley = current_trolley_pos + min_distance
                else:
                    # Fallback: use center of valid range
                    self.set_point_trolley = (valid_min + valid_max) / 2.0
        else:
            self.set_point_trolley = np.random.uniform(valid_min, valid_max)

        if self.hoist_min is None or self.hoist_max is None:
            return

        self.set_point_hoist = np.random.uniform(
            self.hoist_min + 0.1 * (self.hoist_max - self.hoist_min),
            self.hoist_max - 0.1 * (self.hoist_max - self.hoist_min),
        )

    def _state_dict_from_autocrane_message(self, message: dict) -> AutocraneState:
        """
        Converts a dictionary received from the Autocrane system into an
        AutocraneState object. This is necessary because the "channels"
        differ slightly between the autocrane core and the reinforcemen
        learning environment.
        """
        state = {}

        # update limits

        self.gantry_min = float(message["gantry_pos_min"])
        self.gantry_max = float(message["gantry_pos_max"])

        # 0.20 m buffer wasn't enough, with a lot of sway the grapple can hit the middle column
        trolley_buffer_inside = 1.0
        trolley_buffer_outside = 0.50
        self.trolley_min = float(message["trolley_pos_min"]) + trolley_buffer_inside
        self.trolley_max = float(message["trolley_pos_max"]) - trolley_buffer_outside
        if self.trolley_min > self.trolley_max:
            raise ValueError("Trolley min is greater than trolley max")

        hoist_buffer = 0.25
        self.hoist_min = float(message["hoist_pos_min"]) + hoist_buffer
        self.hoist_max = float(message["hoist_pos_max"]) - hoist_buffer
        if self.hoist_min > self.hoist_max:
            raise ValueError("Hoist min is greater than hoist max")

        state["cycle_number"] = message["cycle_number"]
        state["cycle_started_at"] = message["cycle_started_at"]
        state["executed_at"] = message["executed_at"]
        state["active"] = bool(message["crane_active"]) and bool(
            message["behaviour_active"]
        )
        state["gantry_pos"] = float(message["gantry_pos"])
        state["hoist_pos"] = float(message["hoist_pos"])
        state["trolley_pos"] = float(message["trolley_pos"])
        state["grapple_pos"] = float(message["grapple_pos"])
        state["gantry_vel"] = float(message["gantry_vel"])
        state["hoist_vel"] = float(message["hoist_vel"])
        state["trolley_vel"] = float(message["trolley_vel"])
        state["grapple_vel"] = float(message["grapple_vel"])
        state["grapple_sway_trolley"] = float(message["grapple_sway_trolley"])
        state["grapple_sway_gantry"] = float(message["grapple_sway_gantry"])
        state["grapple_loaded"] = bool(message["grapple_loaded"])
        state["gantry_limit_dist_left"] = 0.0
        state["gantry_limit_dist_right"] = 0.0
        state["trolley_limit_dist_left"] = 0.0
        state["trolley_limit_dist_right"] = 0.0
        state["hoist_limit_dist_up"] = 0.0
        state["hoist_limit_dist_down"] = 0.0
        state["gantry_set_point_delta"] = 0.0
        state["trolley_set_point_delta"] = 0.0
        state["hoist_set_point_delta"] = 0.0
        state["gantry_target_vel_ACT"] = 0.0
        state["hoist_target_vel_ACT"] = 0.0
        state["trolley_target_vel_ACT"] = 0.0
        state["grapple_target_vel_ACT"] = 0.0
        state["weight"] = float(message["weight"])

        return state

    def _action_dict_from_action(
        self, action: AutocraneAction, cycle_number: int
    ) -> dict:
        dict = {
            "cycle_number": cycle_number,
            "gantry_target_vel": 0.0,
            "trolley_target_vel": 0.0,
            "hoist_target_vel": 0.0,
        }

        if "gantry_target_vel" in action.channels:
            dict["gantry_target_vel"] = action["gantry_target_vel"]
        if "trolley_target_vel" in action.channels:
            dict["trolley_target_vel"] = action["trolley_target_vel"]
        if self.hoist_active and "hoist_target_vel" in action.channels:
            dict["hoist_target_vel"] = action["hoist_target_vel"]

        return dict

    def _process_and_update_from(
        self, state_dict, action_dict, current_state=None
    ) -> AutocraneState:
        """
        Processes the state dictionary received from the Autocrane system and
        updates it to be compatible with the reinforcement learning environment.
        Also transforms the trolley position and makes limits relative to the
        current set point and updates last cycle number.
        """
        self._last_cycle_number = state_dict["cycle_number"]

        gantry_pos = state_dict["gantry_pos"]
        trolley_pos = state_dict["trolley_pos"]
        hoist_pos = state_dict["hoist_pos"]

        # move absolute position to a relative one in respect to the
        # present set point

        if self._gantry_set_point is not None:
            state_dict["gantry_set_point_delta"] = gantry_pos - self._gantry_set_point

        if self._trolley_set_point is not None:
            state_dict["trolley_set_point_delta"] = (
                trolley_pos - self._trolley_set_point
            )

        if self._hoist_set_point is not None:
            state_dict["hoist_set_point_delta"] = hoist_pos - self._hoist_set_point

        # make limits relative

        state_dict["gantry_limit_dist_left"] = self.gantry_min - gantry_pos
        state_dict["gantry_limit_dist_right"] = self.gantry_max - gantry_pos

        state_dict["trolley_limit_dist_left"] = self.trolley_min - trolley_pos
        state_dict["trolley_limit_dist_right"] = self.trolley_max - trolley_pos

        state_dict["hoist_limit_dist_up"] = self.hoist_max - hoist_pos
        state_dict["hoist_limit_dist_down"] = self.hoist_min - hoist_pos

        # copy current actions to state representation
        state_dict["trolley_target_vel_ACT"] = (
            action_dict["trolley_target_vel"]
            if "trolley_target_vel" in action_dict
            else 0.0
        )
        state_dict["hoist_target_vel_ACT"] = (
            action_dict["hoist_target_vel"]
            if "hoist_target_vel" in action_dict
            else 0.0
        )
        state_dict["gantry_target_vel_ACT"] = (
            action_dict["gantry_target_vel"]
            if "gantry_target_vel" in action_dict
            else 0.0
        )
        state_dict["grapple_target_vel_ACT"] = (
            action_dict["grapple_target_vel"]
            if "grapple_target_vel" in action_dict
            else 0.0
        )

        # calculate grapple sway velocity
        if current_state is not None:
            state_dict["grapple_sway_trolley_vel"] = (
                current_state["grapple_sway_trolley"]
                - state_dict["grapple_sway_trolley"]
            )
            state_dict["grapple_sway_gantry_vel"] = (
                current_state["grapple_sway_gantry"] - state_dict["grapple_sway_gantry"]
            )
            # TODO: normalize by time to fight jitter
        else:
            state_dict["grapple_sway_trolley_vel"] = 0.0
            state_dict["grapple_sway_gantry_vel"] = 0.0

        # print("State dict:", state_dict)
        # print("Set point:", self.set_point_trolley)

        # Create the new state
        new_state = AutocraneState(state_dict)

        # Check for bad terminal states (limits and excessive sway)
        bad_terminal = False

        if self._axis == "gantry":
            if gantry_pos <= self.gantry_min or gantry_pos >= self.gantry_max:
                print(
                    "ZMQProxy: Gantry limit reached. Terminal state. Gantry position:",
                    gantry_pos,
                    "limits:",
                    self.gantry_min,
                    "-",
                    self.gantry_max,
                )
                new_state.terminal = True
                bad_terminal = True

        if self._axis == "trolley" and (
            trolley_pos <= self.trolley_min or trolley_pos >= self.trolley_max
        ):
            print("ZMQProxy: Trolley limit reached. Terminal state.")
            new_state.terminal = True
            bad_terminal = True

        if hoist_pos <= self.hoist_min or hoist_pos >= self.hoist_max:
            print(
                "ZMQProxy: Hoist limit reached. Terminal state. Hoist position:",
                hoist_pos,
                "limits:",
                self.hoist_min,
                "-",
                self.hoist_max,
            )
            new_state.terminal = True
            bad_terminal = True

        # sway limit (use axis-appropriate sway channel)
        sway_trolley = abs(state_dict["grapple_sway_trolley"])
        sway_gantry = abs(state_dict["grapple_sway_gantry"])
        sway = sway_gantry if self._axis == "gantry" else sway_trolley
        if sway > self._sway_terminal_margin:
            sway_key = (
                "grapple_sway_gantry"
                if self._axis == "gantry"
                else "grapple_sway_trolley"
            )
            print(
                "ZMQProxy: Sway limit reached. Terminal state. Sway:",
                state_dict[sway_key],
            )
            new_state.terminal = True
            bad_terminal = True

        # Check for good terminal state (within margins for consecutive steps)
        if self._axis == "gantry":
            set_point_ok = self._gantry_set_point is not None
            position_delta = abs(state_dict["gantry_set_point_delta"])
        else:
            set_point_ok = self._trolley_set_point is not None
            position_delta = abs(state_dict["trolley_set_point_delta"])

        if not bad_terminal and set_point_ok:
            within_distance = position_delta <= self._good_terminal_distance_margin
            within_sway = sway <= self._good_terminal_sway_margin

            if within_distance and within_sway:
                self._consecutive_good_steps += 1
                if (
                    self._consecutive_good_steps
                    >= self._good_terminal_consecutive_steps
                ):
                    print(
                        f"ZMQProxy: Good terminal state reached. Within margins for {self._consecutive_good_steps} steps. Distance: {position_delta:.4f}, Sway: {sway:.4f}"
                    )
                    new_state.terminal = True
                    self._is_good_terminal = True
            else:
                self._consecutive_good_steps = 0
                self._is_good_terminal = False
        else:
            self._consecutive_good_steps = 0
            self._is_good_terminal = False

        return new_state

    def _receive_message(self) -> dict:
        """
        Receives the latest message from the Autocrane system. Will read all available messages until none are left and return only the latest one.
        If no message is available yet, it will block and wait for one to arrive

        Returning only the latest message ensures synchronization between core and proxy, even when the proxy is inactive (e.g., during controller updates). This should not happen during loop execution, because here, in that situation, it would cause information loss, destroy transition integrity, and cause jitter. Autocrane core will notice that the proxy is not repsoning in time and will warn about it in the logfiles. We have not duplicated a warning mechanism here on this side. Thus, check the autocrane core logs to make sure this does not happen with your experimental setup.
        """
        latest_message = None
        while True:
            try:
                message = self.state_socket.recv_json(flags=zmq.NOBLOCK)
                latest_message = message
            except zmq.Again:  # no message received yet
                if latest_message is not None:
                    break
                continue
        return latest_message

    def _get_next_state(
        self, state: AutocraneState, action: AutocraneAction
    ) -> AutocraneState:
        """
        Sends an action to the Autocrane system and receives the next state.
        """

        # Send the action to the autocrane agent
        action_dict = self._action_dict_from_action(action, self._last_cycle_number)
        # print("Sending action to autocrane agent:", action_dict)
        self.action_socket.send_json(action_dict)

        # Receive the next state from the autocrane agent
        latest_message = self._receive_message()
        state_dict = self._state_dict_from_autocrane_message(latest_message)

        new_state = self._process_and_update_from(
            state_dict, action_dict, self._current_state
        )

        if self._axis == "gantry":
            print(
                "\t L: {:.2f}".format(new_state["gantry_limit_dist_left"]),
                "\t G: {:.2f}".format(new_state["gantry_set_point_delta"]),
                "\t R: {:.2f}".format(new_state["gantry_limit_dist_right"]),
                "\t AG: {:.3f}".format(action_dict["gantry_target_vel"]),
                "\t\tU: {:.2f}".format(new_state["hoist_limit_dist_up"]),
                "\t H:  {:.2f}".format(new_state["hoist_set_point_delta"]),
                "\t D: {:.2f}".format(new_state["hoist_limit_dist_down"]),
                "\t AH: {:.3f}".format(action_dict["hoist_target_vel"]),
            )
        else:
            print(
                "\t L: {:.2f}".format(new_state["trolley_limit_dist_left"]),
                "\t T: {:.2f}".format(new_state["trolley_set_point_delta"]),
                "\t R: {:.2f}".format(new_state["trolley_limit_dist_right"]),
                "\t AT: {:.3f}".format(action_dict["trolley_target_vel"]),
                "\t\tU: {:.2f}".format(new_state["hoist_limit_dist_up"]),
                "\t H:  {:.2f}".format(new_state["hoist_set_point_delta"]),
                "\t D: {:.2f}".format(new_state["hoist_limit_dist_down"]),
                "\t AH: {:.3f}".format(action_dict["hoist_target_vel"]),
            )
        self._current_state = new_state
        return new_state

    def get_next_state(
        self, state: AutocraneState, action: AutocraneAction
    ) -> AutocraneState:
        """
        Override to ensure terminal states have correct costs:
        - Bad terminal states (limits, excessive sway): cost = 1.0
        - Good terminal states (within margins): cost = 0.0
        """
        new_state = super().get_next_state(state, action)

        # Override cost for terminal states
        if new_state.terminal:
            if self._is_good_terminal:
                # Good terminal: cost = 0.0
                new_state.cost = 0.0
            else:
                # Bad terminal: cost = 1.0
                new_state.cost = 1.0

        return new_state

    def check_initial_state(self, state: Optional[AutocraneState]) -> AutocraneState:
        """
        Checks the initial state of the Autocrane system. Extracts the limits, selects the very first random set point, if this feature is enabled and
        sets a proper current state for the episode to start.
        """

        # Receive the initial state information from the autocrane agent
        latest_message = self._receive_message()
        print("Received initial state from autocrane agent:", latest_message)
        state_dict = self._state_dict_from_autocrane_message(latest_message)

        if self._axis == "gantry":
            if self._gantry_set_point is None and self._randomize_set_points:
                self.set_random_set_point()
            if self._gantry_set_point is None and self._alternating_set_points:
                self.set_alternating_set_point()
        else:
            if self._trolley_set_point is None and self._randomize_set_points:
                self.set_random_set_point()
            if self._trolley_set_point is None and self._alternating_set_points:
                self.set_alternating_set_point()

        # Create the initial state
        initial_state = self._process_and_update_from(
            state_dict,
            {
                "trolley_target_vel": 0.0,
                "hoist_target_vel": 0.0,
                "gantry_target_vel": 0.0,
            },
        )

        self._current_state = initial_state
        return initial_state

    def notify_episode_starts(self) -> bool:
        """
        Callback function that is called when a new episode is started. Moves away from the limits, if the crane is presently to close. Sets a new random set point, if this feature is enabled.
        """
        super().notify_episode_starts()

        # Reset good terminal tracking
        self._consecutive_good_steps = 0
        self._is_good_terminal = False

        if self._randomize_set_points:
            self.set_random_set_point()

        elif self._alternating_set_points:
            self.set_alternating_set_point()

        self.move_away_from_limits()
        return True

    def notify_episode_stops(self) -> bool:
        """
        Callback function that is called when an episode is stopped. Presently does nothing.
        """
        return True

    def move_away_from_limits(self):
        """
        Moves the controlled axis (trolley or gantry) away from the limits if too close.
        Distance should be at least 0.10 m (trolley) or 0.10 m (gantry) from limits.
        """
        target_trolley_vel = 0.0
        target_gantry_vel = 0.0
        target_hoist_vel = 0.0

        while True:
            message = self._receive_message()

            trolley_pos = float(message["trolley_pos"])
            gantry_pos = float(message["gantry_pos"])
            hoist_pos = float(message["hoist_pos"])

            state_dict = self._state_dict_from_autocrane_message(message)
            self._process_and_update_from(
                state_dict,
                {
                    "gantry_target_vel": target_gantry_vel,
                    "trolley_target_vel": target_trolley_vel,
                    "hoist_target_vel": target_hoist_vel,
                },
            )
            cycle_number = state_dict["cycle_number"]

            if self._axis == "gantry":
                print(
                    f"IN MOVE AWAY FROM LIMITS: Cycle {cycle_number} Gantry position: {gantry_pos}, limits: {self.gantry_min} - {self.gantry_max}, hoist: {self.hoist_min}-{self.hoist_max}"
                )
            else:
                print(
                    f"IN MOVE AWAY FROM LIMITS: Cycle {cycle_number} Trolley position: {trolley_pos}, limits: {self.trolley_min} - {self.trolley_max}, {self.hoist_min}-{self.hoist_max}"
                )

            if self._axis == "gantry":
                if self.gantry_min is None or self.gantry_max is None:
                    gantry_ok = True
                else:
                    gantry_ok = (
                        gantry_pos > self.gantry_min + 0.10
                        and gantry_pos < self.gantry_max - 0.10
                    )
                trolley_ok = True
            else:
                if self.trolley_min is None or self.trolley_max is None:
                    trolley_ok = True
                else:
                    trolley_ok = (
                        trolley_pos > self.trolley_min + 0.10
                        and trolley_pos < self.trolley_max - 0.1
                    )
                gantry_ok = True

            if self.hoist_min is None or self.hoist_max is None:
                hoist_ok = True
            else:
                hoist_ok = not self.hoist_active or (
                    self.hoist_active
                    and hoist_pos > self.hoist_min + 0.20
                    and hoist_pos < self.hoist_max - 0.20
                )

            if trolley_ok and gantry_ok and hoist_ok:
                print("Axis and hoist are ok, moving away from limits")
                break

            if self._axis == "gantry":
                if (
                    not gantry_ok
                    and self.gantry_min is not None
                    and self.gantry_max is not None
                ):
                    center = (self.gantry_min + self.gantry_max) / 2.0
                    target_gantry_vel = 1.0 if gantry_pos <= center else -1.0
                else:
                    target_gantry_vel = 0.0
            else:
                if (
                    not trolley_ok
                    and self.trolley_min is not None
                    and self.trolley_max is not None
                ):
                    center = (self.trolley_min + self.trolley_max) / 2.0
                    target_trolley_vel = 0.268 if trolley_pos <= center else -0.268
                else:
                    target_trolley_vel = 0.0

            if (
                not hoist_ok
                and self.hoist_min is not None
                and self.hoist_max is not None
            ):
                center = (self.hoist_min + self.hoist_max) / 2.0
                target_hoist_vel = 0.095 if hoist_pos <= center else -0.095
            else:
                target_hoist_vel = 0.0

            self.action_socket.send_json(
                {
                    "gantry_target_vel": target_gantry_vel,
                    "trolley_target_vel": target_trolley_vel,
                    "hoist_target_vel": target_hoist_vel,
                    "cycle_number": cycle_number,
                }
            )

    def __del__(self):
        self.state_socket.close()
        self.action_socket.close()
        self.context.term()


if __name__ == "__main__":
    # Example usage

    print("Starting plant")
    plant = AutocraneZMQProxyPlant(randomize_set_points=True)
    print("Plant started")

    # Simulate an episode
    plant.notify_episode_starts()

    print("Checking initial state")
    state = plant.check_initial_state(None)
    print("Initial state:", state)

    print("Driving towards set point: ", plant.set_point_trolley)

    for _ in range(400):  # run some steps and drive towards the set point
        print("Step", _)

        if state["trolley_pos"] > 0:
            target_trolley_vel = -0.268
        else:
            target_trolley_vel = 0.268

        action = AutocraneDiscreteAction(
            {
                "gantry_target_vel": 0.0,
                "trolley_target_vel": target_trolley_vel,
                "hoist_target_vel": 0.0,
            }
        )
        state = plant.get_next_state(plant._current_state, action)
        print("Action:", action)
        print("Next state:", state)

    plant.notify_episode_stops()
