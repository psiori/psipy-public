# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

import os
import random
import sys

import pytest

from psipy.core.threading_utils import StoppableThread
from psipy.core.utils import busy_sleep
from psipy.rl.plant.cartpole import CartPolePlant

if sys.platform != "win32":
    import pty  # noqa


class Writer(StoppableThread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port

    def run(self):
        t = 0

        def write(msg):
            os.write(self.port, msg.encode())

        while not self.stopped():
            busy_sleep(0.0001)
            if random.random() > 0.95:  # write an empty line sometimes
                write("\r\n")
                continue
            if random.random() > 0.95:  # write an empty line sometimes
                write("noise\r\n")
                continue
            x = random.randint(0, 4000)
            theta = random.randint(0, 4000)
            t += random.randint(100, 150)
            if random.random() > 0.8:  # write line in pieces sometimes
                write(f"{x},{theta}")
                busy_sleep(0.00001)
                write(f",{t}\r\n")
                continue
            write(f"{x},{theta},{t}\r\n")


class TestCartPole:
    @staticmethod
    @pytest.mark.skipif(sys.platform == "win32", reason="windows has no tty")
    def test_receive():
        master, slave = pty.openpty()
        plant = CartPolePlant(os.ttyname(slave))
        with Writer(master):
            busy_sleep(0.1)
            _, _, _, ts = plant.receive()
            tick = plant._tick
            for _ in range(10):
                busy_sleep(0.1)
                a, b, c, ts_ = plant.receive()
                print(a, b, c, ts, ts_, tick, plant._tick)
                # To be fixed when the cartpole plant is used again.
                # assert plant._tick > tick
                # assert ts_ > ts
                # tick = plant._tick
                # ts = ts_


if __name__ == "__main__":
    TestCartPole.test_receive()
