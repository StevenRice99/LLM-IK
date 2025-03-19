import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d1 = 0.1197
    d2 = 0.093
    d3 = 0.09465
    d4 = 0.0823
    wrist_y = y - d4
    wrist_x = x
    wrist_z = z
    theta1 = math.atan2(wrist_x, wrist_z)
    r = math.sqrt(wrist_x ** 2 + wrist_z ** 2)
    y_adj = wrist_y + d1 - d2
    D = math.sqrt(r ** 2 + y_adj ** 2)
    cos_theta2 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    cross_product = r * (L1 + L2 * math.cos(theta2)) - y_adj * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(y_adj, r)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta2_final = alpha - beta
    theta3 = -theta2
    theta4 = 0
    theta5 = 0
    return (theta1, theta2_final, theta3, theta4, theta5)