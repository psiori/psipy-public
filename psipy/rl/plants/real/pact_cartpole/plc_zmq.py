# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""ZMQ Communication to C Hamburg Cartpole Hardware Script.

To connect, make sure you are on 192.168.0.xxx, where xxx is not 0, 1, 2, or 100.
"""
import logging
from typing import Optional, Tuple, Union

import zmq

LOG = logging.getLogger(__name__)


class Commands:
    """Mapping of readable commands to ints.

    To make more move blocks, do the following:

        1. Add a SET_SPEEDx and two EXEC_SPEEDx (since every added move block sets
           the positive and negative version of the given speed except for SET_SPEED,
           which sets left, right, and stop
        2. Add the new SET_SPEEDx command to SET_SPEED_COMMANDS.
        3. Add the new EXEC_SPEEDx commands to :meth:`execution_registry`.

    """

    NOOP = -1
    MOTOR_ON = 0
    MOTOR_OFF = 1
    RESET_ON = 2
    RESET_OFF = 3
    CALIBRATE = 4
    CENTER = 6
    SHUTDOWN = 15
    RESET_ALL = 17
    HERTZ = 18  # Default = 30Hz
    CONT_SPEED = 20  # Continuous (provide a speed) and activate
    SET_SPEED = 14  # Set first three speeds (e.g. left, stop, and right)
    SET_SPEED2 = 23  # Set second double set of speeds (21 & 22)
    SET_SPEED3 = 24  # Set third double set of speeds (26 & 27)
    # SET_SPEED4 = 25  # Set fourth double set of speeds (28 & 29)
    EXEC_SPEED0 = 8  # Speed -1 / continuous
    EXEC_SPEED1 = 9  # Speed 0
    EXEC_SPEED2 = 10  # Speed 1
    EXEC_SPEED3 = 21  # Speed -2
    EXEC_SPEED4 = 22  # Speed 2
    EXEC_SPEED5 = 26  # Speed -3
    EXEC_SPEED6 = 27  # Speed 3
    # EXEC_SPEED7 = ?
    # EXEC_SPEED8 = ?
    RESET_EXECUTES = 28
    RELINQUISH_CONTROL = 29

    @classmethod
    def SET_SPEED_COMMANDS(cls) -> Tuple[int, ...]:
        """All set speed commands in a tuple"""
        return cls.SET_SPEED, cls.SET_SPEED2, cls.SET_SPEED3  # , cls.SET_SPEED4


class HardwareComms:
    """Handles message send/receive to the Hilscher card.

    Available hosts:

        * 127.0.0.1:5555 (Hamburg VPN for Windows)
        * pact-one.localdomain:5555 (Pact One)

    """

    def __init__(self, hostname: str, port: int = 5555):
        self.hostname = hostname
        self.port = port
        self.zmq = zmq.Context()
        self.active = False
        self.socket = None

    def send_NOOP(self):
        """Convenience function for sending the NOOP"""
        self.send(Commands.NOOP)

    def send(
        self,
        command: int,
        speed: Union[int, float, str] = "",
        cost: Union[float, str] = "",
        q: Union[float, str] = "",
    ):
        """Send a command and optional speed to the card.

        Data sent is a stringified int. Extra information can be sent, separated
        by colons. The available extra information to send is: speed, cost, and
        q value.

        If sending extra info, the format is:

            command:speed:cost:q

        If any information is missing, the space for the value will be empty, e.g.

            command::cost:
        """
        # Round values; can't round in packed_command because they can be strings
        if cost:
            cost = f"{cost:.4f}"
        if q:
            q = f"{q:.4f}"
        packed_command = f"{command}:{speed}:{cost}:{q}"
        LOG.debug(f"Sending: {packed_command} ({type(packed_command)})")
        self.socket.send_string(packed_command)

    def receive(self) -> str:
        """Receive from the card."""
        msg = self.socket.recv_string()  # blocking
        LOG.debug(f"Message and type: {msg} ({type(msg)}")
        return msg

    def notify_episode_starts(self):
        """Connect to the socket upon episode start."""
        try:
            LOG.info("Connecting zmq...")
            self.socket = self.zmq.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.hostname}:{self.port}")
        except zmq.ZMQError:
            raise ValueError("Port already taken!")
        self.active = True
        LOG.info(f"Connected hardware comms zmq to {self.hostname}:{self.port}.")

    def notify_episode_stops(self):
        """Disconnect from the socket upon episode stop."""
        self.socket.close()
        # del self.socket # TODO?
        self.active = False
        LOG.info("Closed hardware comms zmq socket.")

    def __del__(self):
        if self.socket is not None:
            self.socket.close()
        self.zmq.destroy()


if __name__ == "__main__":
    print("Starting...")
    comms = HardwareComms("pact-one.localdomain", 5555)
    comms.send(Commands.MOTOR_OFF)
    import time

    for _ in range(10000):
        start = time.time()
        print("Noop sending")
        comms.send(Commands.NOOP)
        print("Receiving")
        comms.receive()
        print("TIME:", time.time() - start)

    # comms.send(Commands.RESET_ALL)
    # comms.receive()
    # comms.send(Commands.MOTOR_ON)
    # comms.receive()
    # comms.send(Commands.CALIBRATE)
    # comms.receive()
    # comms.send(Commands.CENTER)
    # comms.receive()
