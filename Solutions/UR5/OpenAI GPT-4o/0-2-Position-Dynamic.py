import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    L1 = 0.425
    L2 = 0.39225
    y_adjusted = y - d1
    theta1 = math.atan2(y_adjusted, x)
    x_prime = math.cos(-theta1) * x + math.sin(-theta1) * y_adjusted
    z_prime = z
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_prime, x_prime)
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)