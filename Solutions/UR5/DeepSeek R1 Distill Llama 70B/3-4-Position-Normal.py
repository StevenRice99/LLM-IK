import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    p_x, p_y, p_z = p
    A = 0.093
    B = 0.09465
    cos_theta2 = (p_x ** 2 + p_y ** 2 - A ** 2 - B ** 2) / (2 * A * B)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    C = A + B * cos_theta2
    D = B * sin_theta2
    if C == 0 or D == 0:
        raise ValueError('C or D is zero; unable to compute theta1.')
    sin_theta1 = (p_x + p_y) / (2 * C)
    cos_theta1 = (p_x - p_y) / (2 * D)
    norm = math.sqrt(sin_theta1 ** 2 + cos_theta1 ** 2)
    if norm != 0:
        sin_theta1 /= norm
        cos_theta1 /= norm
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)