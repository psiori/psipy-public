# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""Listen in on ZMQ publishers.

By default configured to dump all messages sent from a running loop / cycle
manager pair.
"""

import zmq


if __name__ == "__main__":
    # Socket to talk to server
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.connect("tcp://localhost:5557")
    sub.setsockopt(zmq.CONFLATE, True)
    sub.subscribe("")  # subscribe to all topics
    # sub.setsockopt(zmq.SUBSCRIBE, b"cmd")
    # sub.setsockopt(zmq.SUBSCRIBE, b"data")
    # sub.setsockopt(zmq.SUBSCRIBE, b"logs")

    while True:
        topic, msg = sub.recv_multipart()
        print(topic.decode(), msg.decode())
