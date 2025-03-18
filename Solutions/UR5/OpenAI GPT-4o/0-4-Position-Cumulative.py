import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    y2 = y - 0.13585 * math.cos(theta1)
    z2 = z - 0.13585 * math.sin(theta1)
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(y2 ** 2 + z2 ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    alpha = math.atan2(z2, y2)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta3 = alpha - beta
    z3 = z2 - L1 * math.sin(theta3) - L2 * math.sin(theta3 + theta2)
    y3 = y2 - L1 * math.cos(theta3) - L2 * math.cos(theta3 + theta2)
    theta4 = math.atan2(z3, y3)
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)