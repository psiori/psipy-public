import logging
from typing import Optional, Union

import zmq

LOG = logging.getLogger(__name__)


class Commands:
    """Mapping of readable commands to ints"""

    NOOP = -1
    MOTOR_ON = 0
    MOTOR_OFF = 1
    RESET_ON = 2
    RESET_OFF = 3
    CALIBRATE = 4
    CENTER = 6
    EXEC_SPEED0 = 8  # LEFT
    EXEC_SPEED1 = 9  # HALT
    EXEC_SPEED2 = 10  # RIGHT
    SET_SPEED = 14
    SHUTDOWN = 15
    RESET_ALL = 17
    HERTZ = 18  # Default = 30
    LEFT_CONT = 20  # LEFT Continuous (provide a speed)
    RIGHT_CONT = 21  # RIGHT Continuous (provide a speed)
    EXEC_SPEED3 = 22 # LEFT 2nd speed
    EXEC_SPEED4 = 23 # RIGHT 2nd speed
    SET_SPEED2 = 24 # Set second set of speeds (22 & 23)


class HardwareComms:
    """Handles message send/receive to the Hilscher card."""

    def __init__(self, port: str = "5555"):
        self.zmq = zmq.Context()
        self.socket = self.zmq.socket(zmq.REQ)
        try:
            LOG.info("Connecting zmq...")
            self.socket.connect(f"tcp://192.168.0.100:{port}")
        except zmq.ZMQError:
            raise ValueError("Port already taken!")
        LOG.info(f"Connected zmq to port {port}.")

    def send_NOOP(self):
        """Convenience function for sending the NOOP"""
        self.send(Commands.NOOP)

    def send(self, command: int, speed: Optional[Union[int, float]] = None):
        """Send a command and optional speed to the card.

        Data sent is a stringified int. If sending a speed, the format is:

            command:speed
        """
        if speed:
            assert abs(speed)
        packed_command = f"{command}{f':{speed}' if speed else ''}"
        LOG.debug(f"Sending: {packed_command} ({type(packed_command)})")
        self.socket.send_string(packed_command)

    def receive(self) -> str:
        """Receive from the card."""
        msg = self.socket.recv_string()  # blocking
        LOG.debug(f"Message and type: {msg} ({type(msg)}")
        return msg

    def __del__(self):
        self.socket.close()
        self.zmq.destroy()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("port", help="Port for Hardware ZMQ Coms")
    # port = parser.parse_args().port
    print("Starting...")

    comms = HardwareComms()
    # while True:
    #     start = time.time()
    #     comms.send(Commands.NOOP)
    #     comms.receive()
    #     print("TIME:", time.time() - start)
    # comms.send(Commands.RESET_TAGS)
    # comms.receive()
    comms.send(Commands.MOTOR_OFF)
    comms.receive()
    # comms.send(Commands.CALIBRATE)
    # comms.receive()
    # comms.send(Commands.CENTER)
    # comms.receive()
    # comms.send(Commands.SET_SPEED, 100)
    # comms.receive()
    # comms.send(Commands.EXEC_SPEED0)
    # comms.receive()
    # time.sleep(1)
    # comms.send(Commands.EXEC_SPEED1)
    # comms.receive()
    # time.sleep(1)
    # comms.send(Commands.EXEC_SPEED2)
    # comms.receive()
