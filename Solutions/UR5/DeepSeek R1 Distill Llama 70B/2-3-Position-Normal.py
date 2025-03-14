import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    A = 0.39225
    B = 0.093
    p_x, p_y, p_z = p
    L_squared = p_x ** 2 + p_z ** 2
    numerator = L_squared - A ** 2 - B ** 2
    denominator = 2 * A * B
    cos_C = numerator / denominator if denominator != 0 else 0.0
    cos_C = max(min(cos_C, 1.0), -1.0)
    C = math.acos(cos_C)
    D = A + B * math.cos(C)
    E = B * math.sin(C)
    magnitude = math.sqrt(D ** 2 + E ** 2)
    if magnitude < 1e-09:
        theta1 = 0.0
    else:
        delta = math.atan2(E, D)
        sin_theta = p_x / magnitude
        sin_theta = max(min(sin_theta, 1.0), -1.0)
        theta1 = delta + math.asin(sin_theta)
    theta2 = theta1 - C
    return (theta1, theta2)