import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    theta3 = math.atan2(x, y - L2)
    x_j4 = 0
    y_j4 = L2
    z_j4 = L1 + L3
    x_j4_rot = x_j4 * math.cos(theta3) - y_j4 * math.sin(theta3)
    y_j4_rot = x_j4 * math.sin(theta3) + y_j4 * math.cos(theta3)
    x_tcp = x_j4_rot + L4 * math.sin(theta3)
    y_tcp = y_j4_rot + L4 * math.cos(theta3)
    z_tcp = z_j4
    r = math.sqrt(x ** 2 + (z - L1) ** 2)
    cos_theta2 = (r ** 2 - L1 ** 2 - L3 ** 2) / (2 * L1 * L3)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_gamma = L3 * math.sin(theta2) / r
    sin_gamma = max(min(sin_gamma, 1.0), -1.0)
    gamma = math.asin(sin_gamma)
    phi = math.atan2(x, z - L1)
    theta1 = phi - gamma
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)