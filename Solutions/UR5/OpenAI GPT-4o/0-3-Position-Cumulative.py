import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.093
    y_adj = y - tcp_offset
    z_adj = z
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(y_adj ** 2 + z_adj ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(y_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    theta1 = math.atan2(-x, y)
    theta4 = 0
    return (theta1, theta2, theta3, theta4)