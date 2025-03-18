import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    x_rot = x * cos_theta1 + y * sin_theta1
    y_rot = -x * sin_theta1 + y * cos_theta1
    z_rot = z
    tcp_offset = 0.093
    y_rot_adj = y_rot - tcp_offset
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_rot ** 2 + z_rot ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    cross_product = x_rot * (L1 + L2 * math.cos(theta2)) - z_rot * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x_rot, z_rot)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta3 = alpha - beta
    theta4 = 0
    return (theta1, theta2, theta3, theta4)