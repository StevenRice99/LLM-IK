import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    d1 = 0.13585
    d2 = 0.1197
    d3 = 0.425
    d4 = 0.39225
    d5 = 0.093
    tcp_offset = 0.09465
    theta1 = math.atan2(-px, py)
    x2 = px
    y2 = py - d1 * math.cos(theta1)
    z2 = pz - d1 * math.sin(theta1)
    theta2 = math.atan2(x2, z2)
    x3 = x2
    y3 = y2 - d2 * math.cos(theta2)
    z3 = z2 - d2 * math.sin(theta2)
    theta3 = math.atan2(x3, z3)
    x4 = x3
    y4 = y3
    z4 = z3 - d4
    theta4 = math.atan2(x4, z4)
    x5 = x4
    y5 = y4 - d5 * math.cos(theta4)
    z5 = z4 - d5 * math.sin(theta4)
    theta5 = rz
    return (theta1, theta2, theta3, theta4, theta5)