import numpy as np
from math import atan2, sqrt, acos, pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    l1 = 0.1197
    l2 = 0.39225
    l3 = 0.093
    l4 = 0.09465
    l5 = 0.0823
    theta1 = atan2(y, x)
    x_plane = sqrt(x ** 2 + y ** 2)
    z_plane = z - l1
    D = (x_plane ** 2 + z_plane ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    D = np.clip(D, -1, 1)
    theta3 = acos(D)
    theta2 = atan2(z_plane, x_plane) - atan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta4 = yaw - theta1 - theta2 - theta3
    theta5 = pitch
    return (theta1, theta2, theta3, theta4, theta5)