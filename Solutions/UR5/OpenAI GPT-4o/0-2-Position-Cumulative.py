import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    y_adjusted = y - 0.13585
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x ** 2 + z ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    cross_product = x * (L1 + L2 * math.cos(theta3)) - z * (L2 * math.sin(theta3))
    if cross_product < 0:
        theta3 = -theta3
    alpha = math.atan2(x, z)
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)