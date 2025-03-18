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
    theta1 = math.atan2(x, z)
    x1 = x * math.cos(theta1) + z * math.sin(theta1)
    z1 = -x * math.sin(theta1) + z * math.cos(theta1)
    x2 = x1
    z2 = z1 - 0.39225
    theta2 = math.atan2(x2, z2)
    x3 = x2 * math.cos(theta2) + z2 * math.sin(theta2)
    y3 = y - 0.093
    theta3 = math.atan2(y3, x3)
    x4 = x3 * math.cos(theta3) - y3 * math.sin(theta3)
    z4 = 0.09465
    theta4 = math.atan2(x4, z4)
    return (theta1, theta2, theta3, theta4)