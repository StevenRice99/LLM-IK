import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_target = p[0]
    y_target = p[1]
    z_target = p[2]
    l1 = 0.425
    l2 = 0.39225
    y_offset = -0.1197
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    beta = math.atan2(x_target, z_target)
    alpha = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta - alpha
    return (theta1, theta2)