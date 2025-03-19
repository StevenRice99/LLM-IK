import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, y)
    x_adj = x - 0.093 * math.sin(theta1)
    y_adj = y - 0.093 * math.cos(theta1)
    z_adj = z
    x_2 = x_adj * math.cos(theta1) + y_adj * math.sin(theta1) - 0.13585
    y_2 = -x_adj * math.sin(theta1) + y_adj * math.cos(theta1)
    z_2 = z_adj
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_2 ** 2 + z_2 ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    if z_2 < 0:
        theta3 = -theta3
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    alpha = math.atan2(z_2, x_2)
    theta2 = alpha - beta
    theta4 = 0
    return (theta1, theta2, theta3, theta4)