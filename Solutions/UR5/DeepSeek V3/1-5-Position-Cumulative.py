import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.0823
    theta1 = math.atan2(x, z)
    x_adj = x - tcp_offset * math.cos(theta1)
    z_adj = z - tcp_offset * math.sin(theta1)
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_adj ** 2 + z_adj ** 2)
    d_min = abs(L1 - L2)
    d_max = L1 + L2
    if d < d_min:
        d = d_min
    elif d > d_max:
        d = d_max
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    cross_product = x_adj * (L1 + L2 * math.cos(theta2)) - z_adj * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta3 = alpha - beta - theta1
    theta4 = math.atan2(y, math.sqrt(x_adj ** 2 + z_adj ** 2))
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)