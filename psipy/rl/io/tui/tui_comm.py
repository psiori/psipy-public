from typing import Any, Dict, List, Optional, cast

import zmq

from psipy.core.io import json_decode
from psipy.core.threading_utils import StoppableThread


class TUISubscriber(StoppableThread):
    """The subscriber manages incoming messages in its own thread.

    Keeps track of the latest message per topic, making them available to the
    terminal interface loop. It may happen that individual messages are skipped
    if the terminal interface loop does not read them from this Subscriber
    fast enough. For messages on the ``step`` channel this is on purpose (as the
    primary control loop may run faster than the terminal interface), for other
    topics (like ``exception`` or ``lifecycle``) the terminal interface has to
    be able to degrade accordingly.

    Also it is to note, that messages can only be read once! Once a message is
    read from the subscriber, it is deleted. This is implemented in order to not
    have an event happen twice in quick succession.
    """

    #: Latest conflated messages for individual topics.
    _data: Dict[str, str]

    #: Message buffers for individual topics.
    _buffers: Dict[str, List[Dict[str, str]]]

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port

        self._data = dict()
        self._buffers = dict(logs=[])

        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.subscribe("")  # subscribe to all topics
        self.sub.setsockopt(zmq.RCVTIMEO, 1000)  # wait 1 sec at most per iter
        self.sub.connect(f"tcp://localhost:{port}")

    def __del__(self):
        self.stop()
        self.sub.close()
        self.ctx.destroy()

    def __getitem__(self, topic: str) -> Optional[Dict[str, Any]]:
        if topic == "logs":
            if len(self._buffers[topic]) == 0:
                return None
            return self._buffers[topic].pop(0)
        msg = self._data.pop(topic, "null")  # json_decode("null") == None
        return cast(Optional[Dict[str, Any]], json_decode(msg))

    def run(self):
        while not self.stopped():
            try:
                temp = self.sub.recv_multipart()
            except zmq.ZMQError:
                continue
            try:
                topic, msg = temp
            except ValueError:  # workaround for handling messed up messages
                continue
            topic = topic.decode()
            if topic.startswith("logs."):
                topic, level = topic.split(".", maxsplit=1)
                self._buffers[topic].append(dict(level=level, msg=msg.decode()))
            else:
                self._data[topic] = msg.decode()
