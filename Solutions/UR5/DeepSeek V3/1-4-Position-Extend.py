import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    x_adj = x
    y_adj = y - 0.093
    z_adj = z - 0.09465
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_adj ** 2 + z_adj ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    cross_product = x_adj * (L1 + L2 * math.cos(theta2)) - z_adj * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    theta3 = math.atan2(y_adj, math.sqrt(x_adj ** 2 + z_adj ** 2))
    theta4 = math.atan2(y, x)
    return (theta1, theta2, theta3, theta4)