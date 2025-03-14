import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    y_prime = y * cos_theta1 + z * sin_theta1
    z_prime = -y * sin_theta1 + z * cos_theta1
    L2 = 0.496
    L3 = 0.39225
    D = math.sqrt(y_prime ** 2 + z_prime ** 2)
    numerator = L2 ** 2 + L3 ** 2 - D ** 2
    denominator = 2 * L2 * L3
    if denominator == 0:
        delta = 0.0
    else:
        delta = math.acos(numerator / denominator)
    cos_delta = math.cos(delta)
    sin_delta = math.sin(delta)
    A = L2 + L3 * cos_delta
    B = L3 * sin_delta
    C = L2 - L3 * sin_delta
    D_coeff = -L3 * sin_delta
    det_M = A * C - B * D_coeff
    if det_M == 0:
        cos_phi = 0.0
        sin_phi = 0.0
    else:
        cos_phi = (C * y_prime - B * z_prime) / det_M
        sin_phi = (A * z_prime - D_coeff * y_prime) / det_M
    cos_phi = max(min(cos_phi, 1.0), -1.0)
    sin_phi = max(min(sin_phi, 1.0), -1.0)
    theta2 = math.atan2(sin_phi, cos_phi)
    theta3 = theta2 - delta
    return (theta1, theta2, theta3)