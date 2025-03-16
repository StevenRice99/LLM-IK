import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    x_prime = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - 0.13585
    L2 = 0.425
    L3 = 0.39225
    D = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta2 = (D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    cos_theta3 = (L2 ** 2 + D ** 2 - L3 ** 2) / (2 * L2 * D)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(z_prime, x_prime) - math.atan2(L3 * sin_theta2, L2 + L3 * cos_theta2)
    return (theta1, theta2, theta3)