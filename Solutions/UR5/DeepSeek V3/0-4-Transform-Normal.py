import numpy as np
from math import atan2, acos, sin, cos, sqrt

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    R = np.array([[cos(yaw) * cos(pitch), cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll), cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)], [sin(yaw) * cos(pitch), sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll), sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)], [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll)]])
    tcp_offset = np.array([0, 0, 0.09465])
    w = np.array([x, y, z]) - R @ tcp_offset
    theta1 = atan2(w[1], w[0])
    wx_prime = sqrt(w[0] ** 2 + w[1] ** 2)
    wy_prime = w[2]
    L2 = 0.13585
    L3 = 0.425
    L4 = 0.39225
    D = (wx_prime ** 2 + wy_prime ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    D = max(min(D, 1.0), -1.0)
    theta3 = acos(D)
    theta2 = atan2(wy_prime, wx_prime) - atan2(L3 * sin(theta3), L2 + L3 * cos(theta3))
    theta4 = pitch - theta2 - theta3
    theta5 = yaw - theta1
    return (theta1, theta2, theta3, theta4, theta5)