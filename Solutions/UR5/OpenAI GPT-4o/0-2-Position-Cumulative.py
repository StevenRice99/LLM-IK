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
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta2 = (d ** 2 - 0.425 ** 2 - 0.39225 ** 2) / (2 * 0.425 * 0.39225)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    z_double_prime = z_prime - 0.425 * cos_theta2
    x_double_prime = x_prime - 0.425 * sin_theta2
    theta3 = math.atan2(z_double_prime, x_double_prime)
    return (theta1, theta2, theta3)