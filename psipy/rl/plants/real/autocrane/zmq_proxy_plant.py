import json
import zmq
import numpy as np
from typing import Optional
from psipy.rl.core.plant import Plant, State, Action

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
    _channels = ( # SI units, positions are in meters, velocities are in meters per second
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
        "grapple_sway_gantry",
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
    channels = (
        "trolley_target_vel",
    )
    legal_values = (
        (-0.268, 0.268),    
    )

class AutocraneTrolleyHoistAction(AutocraneAction):
    dtype = "discrete"
    channels = (
        "trolley_target_vel",
        "hoist_taget_vel",
    )
    legal_values = (
        (-0.268, 0.0, 0.268),  
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
            (-1.0, 0.0, 1.0), # max vel is 1.00 m/s
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

    def __init__(self, 
                 state_sub_address: str = "tcp://192.168.50.25:7555", 
                 action_pub_address: str = "tcp://192.168.50.25:7556",
                 randomize_set_points: bool = True,
                 hoist_active: bool = False,
                 action_type: AutocraneAction = None, 
                 **kwargs):
        """
        Initializes the AutocraneZMQProxyPlant.

        Args:
            state_sub_address (str): The address of the ZMQ subscriber socket for receiving states.
            action_pub_address (str): The address of the ZMQ publisher socket for sending actions.
            randomize_set_points (bool): Whether to randomize set points.
        """

        if action_type is not None:
            self.action_type = action_type

        self.hoist_active = hoist_active

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
        self.trolley_min = None
        self.trolley_max = None
        self.hoist_min = None
        self.hoist_max = None

        self._gantry_set_point = None
        self._trolley_set_point = None
        self._hoist_set_point = None

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

    def set_random_set_point(self):
        """
        Draws a random set point for the trolley position from the allowed
        range. Set points will be at least 10% of the movement range away from
        the limits.
        """
        if self.trolley_min is None or self.trolley_max is None:
            return 
        
        self.set_point_trolley = np.random.uniform(
            self.trolley_min + 0.1 * (self.trolley_max - self.trolley_min),
            self.trolley_max - 0.1 * (self.trolley_max - self.trolley_min)
        )

        if self.hoist_min is None or self.hoist_max is None:
            return

        self.set_point_hoist = np.random.uniform(
            self.hoist_min + 0.1 * (self.hoist_max - self.hoist_min),
            self.hoist_max - 0.1 * (self.hoist_max - self.hoist_min)
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
        
        self.trolley_min = float(message["trolley_pos_min"]) + 0.20  # 20 cm buffer
        self.trolley_max = float(message["trolley_pos_max"]) - 0.20  # 20 cm buffer

        self.hoist_min = float(message["hoist_pos_min"]) + 0.25  # 25 cm buffer
        self.hoist_max = float(message["hoist_pos_max"]) - 0.25  # 25 cm buffer

        state["cycle_number"] = message["cycle_number"]
        state["cycle_started_at"] = message["cycle_started_at"]
        state["executed_at"] = message["executed_at"]
        state["active"] = bool(message["crane_active"]) and bool(message["behaviour_active"])
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

    def _action_dict_from_action(self, action: AutocraneAction, cycle_number: int) -> dict:
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
    
    def _process_and_update_from(self, 
                                 state_dict,
                                 action_dict) -> AutocraneState:
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
            state_dict["trolley_set_point_delta"] = trolley_pos - self._trolley_set_point

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
        state_dict["trolley_target_vel_ACT"] = action_dict["trolley_target_vel"] if "trolley_target_vel" in action_dict else 0.0
        state_dict["hoist_target_vel_ACT"] = action_dict["hoist_target_vel"] if "hoist_target_vel" in action_dict else 0.0
        state_dict["gantry_target_vel_ACT"] = action_dict["gantry_target_vel"] if "gantry_target_vel" in action_dict else 0.0

        # print("State dict:", state_dict)
        # print("Set point:", self.set_point_trolley)
        
        # Create the new state
        new_state = AutocraneState(state_dict)

        # TODO: GANTRY LIMITS NOT YES IMPLEMNETED (FOR RADIAL CRANE)
        #if gantry_pos <= self.gantry_min or gantry_pos >= self.gantry_max:
        #    print("ZMQProxy: Gantry limit reached. Terminal state. Gantry position:", gantry_pos, "limits:", self.gantry_min, "-", self.gantry_max)
        #    new_state.terminal = True

        if trolley_pos <= self.trolley_min or trolley_pos >= self.trolley_max:
            print("ZMQProxy: Trolley limit reached. Terminal state.")
            new_state.terminal = True        

        if hoist_pos <= self.hoist_min or hoist_pos >= self.hoist_max:
            print("ZMQProxy: Hoist limit reached. Terminal state. Hoist position:", hoist_pos, "limits:", self.hoist_min, "-", self.hoist_max)
            new_state.terminal = True

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

    def _get_next_state(self, state: AutocraneState, action: AutocraneAction) -> AutocraneState:
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

        new_state = self._process_and_update_from(state_dict, action_dict)

        print("L:\t", new_state["trolley_limit_dist_left"], 
              "\tT:\t", new_state["trolley_set_point_delta"], 
              "\tR:\t", new_state["trolley_limit_dist_right"],
              "\tAT:\t", action_dict["trolley_target_vel"],
              "\t\tU:\t", new_state["hoist_limit_dist_up"],
              "\tH:\t", new_state["hoist_set_point_delta"],
              "\tD:\t", new_state["hoist_limit_dist_down"],
              "\tAH:\t", action_dict["hoist_target_vel"])

        self._current_state = new_state
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

        if self._trolley_set_point is None and self._randomize_set_points:
            self.set_random_set_point()
        
        # Create the initial state
        initial_state = self._process_and_update_from(state_dict, { 
            "trolley_target_vel": 0.0,
            "hoist_target_vel": 0.0,
            "gantry_target_vel": 0.0,
        })

        self._current_state = initial_state
        return initial_state

    def notify_episode_starts(self) -> bool:
        """
        Callback function that is called when a new episode is started. Moves away from the limits, if the crane is presently to close. Sets a new random set point, if this feature is enabled.
        """
        super().notify_episode_starts()

        if self._randomize_set_points:
            self.set_random_set_point()

        self.move_away_from_limits()
        return True

    def notify_episode_stops(self) -> bool:
        """
        Callback function that is called when an episode is stopped. Presently does nothing.
        """
        return True

    def move_away_from_limits(self):
        """
        Moves the trolley away from the limits if it is too close to them. Distance should be at least 0.50 meters.
        """

        target_trolley_vel = 0.0
        target_hoist_vel = 0.0

        while True:
            message = self._receive_message()

            trolley_pos = float(message["trolley_pos"])
            hoist_pos = float(message["hoist_pos"])

            state_dict = self._state_dict_from_autocrane_message(message)
            state = self._process_and_update_from(state_dict, {
                "trolley_target_vel": target_trolley_vel,
                "hoist_target_vel": target_hoist_vel,
            })
            cycle_number = state_dict["cycle_number"]

            print (f"IN MOVE AWAY FROM LIMITS: Cycle { cycle_number } Trolley position: {message['trolley_pos']}, limits: {self.trolley_min} - {self.trolley_max}, {self.hoist_min}-{self.hoist_max}")

            trolley_ok = trolley_pos > self.trolley_min + 1.50 and trolley_pos < self.trolley_max - 1.50

            hoist_ok = not self.hoist_active or (self.hoist_active and hoist_pos > self.hoist_min + 0.20 and hoist_pos < self.hoist_max - 0.20)

            if trolley_ok and hoist_ok:
                print("Trolley and hoist are ok, moving away from limits")
                break

            if not trolley_ok:
                center = (self.trolley_min + self.trolley_max) / 2.0
                target_trolley_vel = 0.268 if trolley_pos <= center else -0.268
            else:
                target_trolley_vel = 0.0

            if not hoist_ok:
                center = (self.hoist_min + self.hoist_max) / 2.0
                target_hoist_vel = 0.095 if hoist_pos <= center else -0.095
            else:
                target_hoist_vel = 0.0

            self.action_socket.send_json({
                "gantry_target_vel": 0.0,
                "trolley_target_vel": target_trolley_vel,
                "hoist_target_vel": target_hoist_vel,
                "cycle_number": cycle_number,
            })

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

        action = AutocraneDiscreteAction({
            "gantry_target_vel": 0.0,
            "trolley_target_vel": target_trolley_vel,
            "hoist_target_vel": 0.0,
        })
        state = plant.get_next_state(plant._current_state, action)
        print("Action:", action)
        print("Next state:", state)

    plant.notify_episode_stops()
