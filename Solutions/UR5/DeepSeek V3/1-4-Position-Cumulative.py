import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.093
    y_adj = y - tcp_offset
    L1 = 0.425
    L2 = 0.39225
    theta1 = math.atan2(x, z)
    d = math.sqrt(x ** 2 + z ** 2)
    D = math.sqrt(d ** 2 + y_adj ** 2)
    cos_theta3 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    sin_theta3 = max(min(sin_theta3, 1), -1)
    alpha = math.atan2(y_adj, d)
    beta = math.atan2(L2 * sin_theta3, L1 + L2 * cos_theta3)
    theta2 = alpha - beta
    theta4 = math.atan2(y_adj, x)
    return (theta1, theta2, theta3, theta4)