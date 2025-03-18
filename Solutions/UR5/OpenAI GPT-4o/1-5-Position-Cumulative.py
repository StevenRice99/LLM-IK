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
    x1 = math.sqrt(x ** 2 + z ** 2)
    z1 = y_adj
    d = math.sqrt(x1 ** 2 + z1 ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    theta2 = math.atan2(z1, x1) - math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    x3 = x1 - (L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3))
    z3 = z1 - (L1 * math.sin(theta2) + L2 * math.sin(theta2 + theta3))
    theta4 = math.atan2(z3, x3)
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)