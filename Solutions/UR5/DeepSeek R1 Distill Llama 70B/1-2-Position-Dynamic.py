import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.425
    B = 0.39225
    cos_theta2 = (x ** 2 + z ** 2 - A ** 2 - B ** 2) / (2 * A * B)
    theta2 = math.acos(cos_theta2)
    C = A + B * math.cos(theta2)
    D = B * math.sin(theta2)
    numerator_sin = z * D - C * x
    denominator_sin = D ** 2 - C ** 2
    sin_theta1 = numerator_sin / denominator_sin
    cos_theta1 = (x - C * sin_theta1) / D
    norm = math.sqrt(sin_theta1 ** 2 + cos_theta1 ** 2)
    if not math.isclose(norm, 1.0, rel_tol=1e-09, abs_tol=1e-09):
        raise ValueError('The target position is not reachable.')
    sin_theta1 /= norm
    cos_theta1 /= norm
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)