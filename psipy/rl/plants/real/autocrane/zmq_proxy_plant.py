import json
import zmq
import numpy as np
from typing import Optional
from psipy.rl.core.plant import Plant, State, Action

"""
    C/C++ side of the crane agent where the state is filled:

    state["cycle_number"] = CYCLE.counter;
    state["cycle_started_at"] = CYCLE.startedAt.time_since_epoch().count();
    state["executed_at"] = t.time_since_epoch().count();

    state["crane_active"] = agent->wm->isCraneActive();
    state["behaviour_active"] = this->active;

    state["gantry_pos"] = agent->crane->gantry->getCurrentPos(t);
    state["hoist_pos"] = agent->crane->hoist->getCurrentPos(t);
    state["trolley_pos"] = agent->crane->trolley->getCurrentPos(t);
    state["grapple_pos"] = agent->crane->grapple->getCurrentPos(t); // rotation

    state["gantry_vel"] = agent->crane->gantry->getCurrentVelocity();
    state["hoist_vel"] = agent->crane->hoist->getCurrentVelocity();
    state["trolley_vel"] = agent->crane->trolley->getCurrentVelocity();
    state["grapple_vel"] = agent->crane->grapple->getCurrentVelocity();

    state["grapple_sway_trolley"] = agent->wm->getSwayAngleTrolley(t); // lateral sway
    state["grapple_sway_gantry"] = agent->wm->getSwayAngleGantry(t); // vertical sway

    state["grapple_loaded"] = agent->crane->grapple->isLoaded(t);

    // CraneSpecifications specs = *agent->crane->specs;
    state["trolley_pos_min"] = agent->crane->trolley->minPos()  ;
    state["trolley_pos_max"] = agent->crane->trolley->maxPos();
    state["trolley_max_speed"] = agent->crane->trolley->getMaxAllowedSpeed();


    From the configuration of the decision making of the minicrane, for the trolley axis:

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
        "trolley_limit_dist_left",
        "trolley_limit_dist_right",
        "gantry_target_vel_ACT",
        "hoist_target_vel_ACT",
        "trolley_target_vel_ACT",
        "grapple_target_vel_ACT",
    )

class AutocraneDiscreteAction(Action):
    dtype = "discrete"
    channels = (
        "trolley_target_vel",
    )
    legal_values = (  # speed is in meters per second
        (
            -0.268,   # max allowed speed
            -0.1,  
            -0.0287,  # close to minimal non-zero speed which would actually cause the very slowest possible movement, about 1/10 of max allowed speed
            0,
            0.0287,
            0.1,
            0.268,
        ),
    )

class AutocraneZMQProxyPlant(Plant[AutocraneState, AutocraneDiscreteAction]):
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
    action_type = AutocraneDiscreteAction

    def __init__(self, 
                 state_sub_address: str = "tcp://192.168.50.25:7555", 
                 action_pub_address: str = "tcp://192.168.50.25:7556",
                 randomize_set_points: bool = True, **kwargs):
        """
        Initializes the AutocraneZMQProxyPlant.

        Args:
            state_sub_address (str): The address of the ZMQ subscriber socket for receiving states.
            action_pub_address (str): The address of the ZMQ publisher socket for sending actions.
            randomize_set_points (bool): Whether to randomize set points.
        """
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
        self._trolley_set_point = None

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

    def _state_dict_from_autocrane_message(self, message: dict) -> AutocraneState:
        """
        Converts a dictionary received from the Autocrane system into an
        AutocraneState object. This is necessary because the "channels"
        differ slightly between the autocrane core and the reinforcemen
        learning environment.
        """
        state = {}

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
        state["trolley_limit_dist_left"] = 0.0
        state["trolley_limit_dist_right"] = 0.0
        state["gantry_target_vel_ACT"] = 0.0
        state["hoist_target_vel_ACT"] = 0.0
        state["trolley_target_vel_ACT"] = 0.0
        state["grapple_target_vel_ACT"] = 0.0

        return state

    def _action_dict_from_action(self, action: AutocraneDiscreteAction, cycle_number: int) -> dict:
        return {
            "trolley_target_vel": action["trolley_target_vel"],
            "cycle_number": cycle_number,
        }
    
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

        raw_trolley_pos = state_dict["trolley_pos"]

        # move absolute position to a relative one in respect to the
        # present set point
        state_dict["trolley_pos"] = state_dict["trolley_pos"] - self.set_point_trolley

        # make limits relative
        state_dict["trolley_limit_dist_left"] = self.trolley_min - raw_trolley_pos
        state_dict["trolley_limit_dist_right"] = self.trolley_max - raw_trolley_pos

        # copy current actions to state representation
        state_dict["trolley_target_vel_ACT"] = action_dict["trolley_target_vel"]

        print("State dict:", state_dict)
        print("Set point:", self.set_point_trolley)
        
        # Create the new state
        new_state = AutocraneState(state_dict)

        if raw_trolley_pos <= self.trolley_min or raw_trolley_pos >= self.trolley_max:
            print("ZMQProxy: Trolley limit reached. Terminal state.")
            new_state.terminal = True        

    def _receive_message(self) -> dict:
        """
        Receives a message from the Autocrane system.
        """
        latest_message = None
        while True:
            try:
                message = self.state_socket.recv_json(flags=zmq.NOBLOCK)
                latest_message = message
            except zmq.Again:
                if latest_message is not None:
                    break
                continue
        return latest_message

    def _get_next_state(self, state: AutocraneState, action: AutocraneDiscreteAction) -> AutocraneState:
        """
        Sends an action to the Autocrane system and receives the next state.
        """

        # Send the action to the autocrane agent
        action_dict = self._action_dict_from_action(action, self._last_cycle_number)
        print("Sending action to autocrane agent:", action_dict)
        self.action_socket.send_json(action_dict)

        # Receive the next state from the autocrane agent
        latest_message = self._receive_message()
        state_dict = self._state_dict_from_autocrane_message(latest_message)

        new_state = self._process_and_update_from(state_dict, action_dict)

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

        # update limits
        self.trolley_min = float(state_dict["trolley_pos_min"]) + 0.20  # 20 cm buffer
        self.trolley_max = float(state_dict["trolley_pos_max"]) - 0.20  # 20 cm buffer

        if self._trolley_set_point is None and self._randomize_set_points:
            self.set_random_set_point()
        
        # Create the initial state
        initial_state = self._process_and_update_from(state_dict, { "trolley_target_vel": 0.0 })

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
        center = (self.trolley_min + self.trolley_max) / 2.0
        target_trolley_vel = 0.0

        while True:
            message = self._receive_message()

            state_dict = self._state_dict_from_autocrane_message(message)
            state = self._process_and_update_from(state_dict, {
                "trolley_target_vel": target_trolley_vel
            })
            cycle_number = state_dict["cycle_number"]

            if state["trolley_pos"] > self.trolley_min + 0.50 and state["trolley_pos"] < self.trolley_max - 0.50:
                break   # done, good enough.

            target_trolley_vel = 0.268 if state["trolley_pos"] <= center else -0.268

            self.action_socket.send_json({
                "trolley_target_vel": target_trolley_vel,
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
            "trolley_target_vel": target_trolley_vel,
        })
        state = plant.get_next_state(plant._current_state, action)
        print("Action:", action)
        print("Next state:", state)

    plant.notify_episode_stops()
