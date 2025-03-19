import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, y)
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    x_1 = c1 * x + s1 * y
    y_1 = -s1 * x + c1 * y
    z_1 = z
    y_offset = 0.01615
    y_1 = y_1 - y_offset
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    x_target = x_1
    z_target = z_1 - l1
    d = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(z_target, x_target)
    gamma = math.atan2(l3 * math.sin(theta3), l2 + l3 * math.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)