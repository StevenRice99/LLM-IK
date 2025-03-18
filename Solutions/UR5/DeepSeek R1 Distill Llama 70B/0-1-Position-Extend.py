import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_d, y_d, z_d = p
    sin_theta2 = (z_d - 0.425) / 0.13585
    theta2 = math.asin(sin_theta2)
    cos_theta2 = math.cos(theta2)
    numerator_sin = y_d - 0.1197 * cos_theta2
    numerator_cos = x_d + 0.1197 * sin_theta2
    denominator = 0.13585 * cos_theta2
    if denominator == 0:
        raise ValueError('cos(theta2) is zero, causing division by zero.')
    sin_theta1 = numerator_sin / denominator
    cos_theta1 = numerator_cos / denominator
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)