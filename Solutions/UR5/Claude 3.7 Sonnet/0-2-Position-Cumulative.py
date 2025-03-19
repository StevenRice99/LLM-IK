import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, y)
    x2 = x * math.cos(theta1) + y * math.sin(theta1) - 0.13585 * math.sin(theta1)
    y2 = -x * math.sin(theta1) + y * math.cos(theta1) - 0.13585 * math.cos(theta1)
    z2 = z
    y2 = y2 - 0.016149999999999998
    L2 = 0.425
    L3 = 0.39225
    r = math.sqrt(x2 ** 2 + z2 ** 2)
    cos_theta3 = (r ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(z2, x2)
    gamma = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)