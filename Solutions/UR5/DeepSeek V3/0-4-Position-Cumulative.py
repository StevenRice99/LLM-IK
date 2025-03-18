import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    x_prime = math.sqrt(x ** 2 + y ** 2)
    y_prime = z - 0.13585
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_prime ** 2 + y_prime ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(y_prime, x_prime)
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    theta4 = -theta2 - theta3
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)