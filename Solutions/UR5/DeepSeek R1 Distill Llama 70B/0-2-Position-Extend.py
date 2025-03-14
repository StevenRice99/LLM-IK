import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1, theta2 = existing_solution((x, y, z))
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    cos_theta2 = math.cos(theta2)
    sin_theta2 = math.sin(theta2)
    x_prime = 0.13585 * cos_theta1 + 0.425 * sin_theta2
    y_prime = 0.13585 * sin_theta1 - 0.1197
    z_prime = -0.13585 * cos_theta1 * sin_theta2 + 0.425 * cos_theta2
    numerator = z * x_prime - x * z_prime
    denominator = z * z_prime + x * x_prime
    if denominator == 0:
        if numerator == 0:
            theta3 = 0.0
        else:
            theta3 = math.pi / 2 if numerator > 0 else -math.pi / 2
    else:
        theta3 = math.atan2(numerator, denominator)
    return (theta1, theta2, theta3)

def existing_solution(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" for the first two joints.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the first two joints to for reaching position "p".
    """
    x, y, z = p
    argument = z / 0.425
    argument_clamped = max(min(argument, 1.0), -1.0)
    theta2 = math.acos(argument_clamped)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    K = 0.425 * sin_theta2
    L = 0.01615
    denominator = K ** 2 + L ** 2
    if denominator == 0:
        cos_theta1 = 0.0
        sin_theta1 = 0.0
    else:
        cos_theta1 = (K * x + L * y) / denominator
        sin_theta1 = (-L * x + K * y) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)