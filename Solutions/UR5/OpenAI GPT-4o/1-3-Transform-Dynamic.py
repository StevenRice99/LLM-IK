import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    d1 = 0.425
    d2 = 0.39225
    tcp_offset = 0.093
    theta1 = math.atan2(px, pz)
    py_adjusted = py - tcp_offset
    r = math.sqrt(px ** 2 + pz ** 2)
    s = py_adjusted - d1
    D = (r ** 2 + s ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    D = max(-1, min(1, D))
    theta2 = math.atan2(s, r) - math.acos(D)
    D3 = (r ** 2 + s ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    D3 = max(-1, min(1, D3))
    theta3 = math.acos(D3)
    theta2 += ry
    theta3 += rz
    return (theta1, theta2, theta3)