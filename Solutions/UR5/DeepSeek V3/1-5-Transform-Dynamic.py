import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = math.atan2(x, z)
    x2 = x * math.cos(theta1) + z * math.sin(theta1)
    y2 = y - 0.1197
    z2 = -x * math.sin(theta1) + z * math.cos(theta1) - 0.425
    theta2 = math.atan2(x2, z2)
    x3 = x2 * math.cos(theta2) + z2 * math.sin(theta2)
    y3 = y2
    z3 = -x2 * math.sin(theta2) + z2 * math.cos(theta2) - 0.39225
    theta3 = math.atan2(x3, z3)
    x4 = x3 * math.cos(theta3) + z3 * math.sin(theta3)
    y4 = y3 - 0.093
    z4 = -x3 * math.sin(theta3) + z3 * math.cos(theta3)
    theta4 = math.atan2(y4, x4)
    x5 = x4 * math.cos(theta4) - y4 * math.sin(theta4)
    y5 = x4 * math.sin(theta4) + y4 * math.cos(theta4)
    z5 = z4 - 0.09465
    theta5 = math.atan2(x5, z5)
    return (theta1, theta2, theta3, theta4, theta5)