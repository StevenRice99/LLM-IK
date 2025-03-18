import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    x_adj = math.sqrt(x ** 2 + y ** 2)
    y_adj = z - 0.13585
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_adj ** 2 + y_adj ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if cos_theta3 < -1:
        cos_theta3 = -1
    elif cos_theta3 > 1:
        cos_theta3 = 1
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(y_adj, x_adj)
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    theta5 = 0.0
    theta6 = 0.0
    return (theta1, theta2, theta3, 0.0, theta5, theta6)