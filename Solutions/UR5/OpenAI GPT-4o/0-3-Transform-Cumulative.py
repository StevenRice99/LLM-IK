import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.093
    theta_1 = math.atan2(y, x)
    wx = x - L4 * math.cos(rz) * math.cos(ry)
    wy = y - L4 * math.sin(rz) * math.cos(ry)
    wz = z - L4 * math.sin(ry)
    d = math.sqrt(wx ** 2 + wy ** 2)
    r = math.sqrt(d ** 2 + (wz - L1) ** 2)
    cos_angle_a = (L2 ** 2 + r ** 2 - L3 ** 2) / (2 * L2 * r)
    angle_a = math.acos(np.clip(cos_angle_a, -1.0, 1.0))
    cos_angle_b = (L2 ** 2 + L3 ** 2 - r ** 2) / (2 * L2 * L3)
    angle_b = math.acos(np.clip(cos_angle_b, -1.0, 1.0))
    theta_2 = math.atan2(wz - L1, d) - angle_a
    theta_3 = math.pi - angle_b
    theta_4 = rz - theta_2 - theta_3
    return (theta_1, theta_2, theta_3, theta_4)