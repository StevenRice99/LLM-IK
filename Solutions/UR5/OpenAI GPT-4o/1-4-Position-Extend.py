import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.09465
    link4_offset = 0.093
    adjusted_y = y - link4_offset
    adjusted_z = z - tcp_offset
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x ** 2 + adjusted_z ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    cross_product = x * (L1 + L2 * math.cos(theta2)) - adjusted_z * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x, adjusted_z)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    theta3 = math.atan2(adjusted_y, math.sqrt(x ** 2 + adjusted_z ** 2))
    theta4 = 0
    return (theta1, theta2, theta3, theta4)