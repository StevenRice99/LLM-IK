import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d4 = 0.093
    d5 = 0.09465
    tcp_offset = 0.0823
    y_adj = y - tcp_offset
    theta1 = math.atan2(x, z)
    r = math.sqrt(y_adj ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta3 = math.atan2(sin_theta2, cos_theta2)
    theta4 = math.atan2(-x, y_adj)
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)