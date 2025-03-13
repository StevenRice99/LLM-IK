import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    a = 0.425
    b = 0.39225
    c = 0.13585
    d = 0.1197
    cos_theta1 = -(y + d) / c
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    theta1 = math.acos(cos_theta1)
    sin_theta1 = math.sqrt(1 - cos_theta1 ** 2) if theta1 >= 0 else -math.sqrt(1 - cos_theta1 ** 2)
    A = x - c * sin_theta1
    M = a
    N = b
    C = (A ** 2 + z ** 2 + M ** 2 - N ** 2) / (2 * M)
    discriminant = (2 * z * C) ** 2 - 4 * (M ** 2 + z ** 2 - C ** 2) * (M ** 2 + z ** 2)
    if discriminant < 0:
        raise ValueError('Negative discriminant, target may be unreachable.')
    sqrt_discriminant = math.sqrt(discriminant)
    u_numerator = 2 * z * C + sqrt_discriminant
    u_denominator = 2 * (M ** 2 + z ** 2 - C ** 2)
    u = u_numerator / u_denominator
    u = max(min(u, 1.0), -1.0)
    theta2 = math.acos(u)
    sin_theta3 = (A - M * math.sin(theta2)) / N
    sin_theta3 = max(min(sin_theta3, 1.0), -1.0)
    theta3 = math.asin(sin_theta3)
    return (theta1, theta2, theta3)