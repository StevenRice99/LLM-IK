import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    d4 = 0.093
    z_adj = z - L3
    y_adj = y - d4
    theta1 = math.atan2(x, z_adj)
    d = math.sqrt(x ** 2 + z_adj ** 2)
    r = math.sqrt(d ** 2 + y_adj ** 2)
    cos_theta3 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta2 = math.atan2(y_adj, d) - math.atan2(L2 * sin_theta3, L1 + L2 * cos_theta3)
    theta4 = math.atan2(y, x)
    return (theta1, theta2, theta3, theta4)