import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    if x_target == 0 and z_target == 0:
        theta1 = 0
    else:
        theta1 = math.atan2(x_target, z_target)
    y_adj = y_target + 0.093
    r_target = math.sqrt(x_target ** 2 + z_target ** 2)
    l1 = 0.425
    l2 = 0.39225
    y_offset = -0.1197
    y_adj = y_adj - y_offset
    d = math.sqrt(r_target ** 2 + y_adj ** 2)
    cos_theta3 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(y_adj, r_target)
    cos_alpha = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta2 = beta - alpha
    return (theta1, theta2, theta3)