# PSIORI Machine Learning Toolbox
# ===========================================
#
# Copyright (C) PSIORI GmbH, Germany
# Proprietary and confidential, all rights reserved.

"""IMU WIFI Connection

"""

import math
import socket
import time
from struct import unpack

import numpy as np

from psipy.core.threading_utils import StoppableThread


class IMUConnection(StoppableThread):
    def __init__(self, UDP_IP):
        super().__init__()
        self.UDP_IP = UDP_IP
        self.connect()

        self.ts = 0
        self.cycle = 0
        self.q0 = 0.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0

        self.buffer_size = 10
        self.output_buffer = np.zeros(self.buffer_size)

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.yaw_dot = 0.0
        self.pitch_dot = 0.0
        self.roll_dot = 0.0

    def run(self):
        while not self.stopped():
            # receive with blocking
            incoming = self.sock.recvfrom(1024)
            data = unpack("<QIfffffffffffffffff", incoming[0])
            self.ts, self.cycle, self.q0, self.q1, self.q2, self.q3 = data[:6]
            ax, ay, az, gx, gy, gz, mx, my, mz, temp, lat, longi, alt = data[6:]
            # gx, gy == 0, 0
            self.ax = ax
            self.ay = ay
            self.gz = gz
            # print("a", ax, ay, az)
            # print("g", gx, gy, gz)
            # a 0.9999119639396667 -0.0014639999717473984 0.0
            # g 0.0 0.0 -0.017500000074505806
            self.calc_ypr_dot()

    def stop(self):
        super().stop()
        self.disconnect()

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # send identify message
        MESSAGE = '{command:"Identify"}'

        if MESSAGE:
            print("Sending start packet to %s: %s" % (self.UDP_IP, MESSAGE))
            self.sock.sendto(MESSAGE.encode(), (self.UDP_IP, 3333))
        else:
            print("Empty messages!! Nothing to write...")

        d = self.sock.recvfrom(1024)
        data = d[0]
        print(data)

        # send status message
        MESSAGE = '{command:"Status"}'
        print("Sending start packet to %s: %s" % (self.UDP_IP, MESSAGE))
        self.sock.sendto(MESSAGE.encode(), (self.UDP_IP, 3333))

        d = self.sock.recvfrom(1024)
        data = d[0]
        print(data)

        MESSAGE = '{command:"BeginSensorTransmission", "time":"%u"}' % (
            time.time() * 1000
        )
        print("Sending start packet to %s: %s" % (self.UDP_IP, MESSAGE))
        self.sock.sendto(MESSAGE.encode(), (self.UDP_IP, 3333))

        if data:
            return True
        else:
            return False

    def disconnect(self):
        MESSAGE = '{command:"EndSensorTransmission", "time":"%u"}' % (
            time.time() * 1000
        )
        print("Sending stop packet to %s: %s" % (self.UDP_IP, MESSAGE))
        self.sock.sendto(MESSAGE.encode(), (self.UDP_IP, 3333))

    @staticmethod
    def quat_to_ypr(q0, q1, q2, q3):
        yaw = math.atan2(q0 * q1 + q2 * q3, 0.5 - q1 * q1 - q2 * q2)
        pitch = math.asin(-2.0 * (q1 * q3 - q0 * q2))
        roll = math.atan2(q1 * q2 + q0 * q3, 0.5 - q2 * q2 - q3 * q3)
        pitch *= 180.0 / math.pi
        yaw *= 180.0 / math.pi
        roll *= 180.0 / math.pi
        return [yaw, pitch, roll]

    def calc_ypr_dot(self):
        yaw, pitch, roll = self.quat_to_ypr(self.q0, self.q1, self.q2, self.q3)
        pitch_moving_average = np.mean(self.output_buffer)
        self.output_buffer = np.roll(self.output_buffer, 1)
        self.output_buffer[0] = pitch
        self.yaw_dot = (abs(self.yaw) - abs(yaw)) / 0.02
        self.pitch_dot = (np.mean(self.output_buffer) - pitch_moving_average) / 0.02
        self.roll_dot = (abs(self.roll) - abs(roll)) / 0.02
        self.yaw, self.pitch, self.roll = yaw, pitch, roll

    def get_euler(self):
        return self.ts, self.yaw, self.pitch, self.roll

    def get_euler_dot(self):
        return (
            self.ts,
            self.yaw,
            self.pitch,
            self.roll,
            self.yaw_dot,
            self.pitch_dot,
            self.roll_dot,
        )
