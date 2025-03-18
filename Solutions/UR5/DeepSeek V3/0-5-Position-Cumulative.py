import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    alpha = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta3 = alpha - beta
    theta4 = math.atan2(math.sin(theta1) * x - math.cos(theta1) * y, z)
    theta5 = math.atan2(math.sqrt(x ** 2 + y ** 2), z)
    theta6 = math.atan2(math.cos(theta1) * x + math.sin(theta1) * y, z)
    return (theta1, theta2, theta3, theta4, theta5, theta6)