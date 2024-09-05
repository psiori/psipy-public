"""ZMQ example server.

Complements `zmq/server.py`. Start each in its own terminal to play around
with zmq pub/sub communication.
"""

import random
import sys
import time

import zmq

if __name__ == '__main__':
    port = "5556"
    if len(sys.argv) > 1:
        port = sys.argv[1]
        int(port)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)

    while True:
        topic = random.randrange(9999, 10005)
        messagedata = random.randrange(1, 215) - 80
        print("%d %d" % (topic, messagedata))
        socket.send(("%d %d" % (topic, messagedata)).encode("utf-8"))
        time.sleep(1)
