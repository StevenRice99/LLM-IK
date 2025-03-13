import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    y_local = -x * sin_theta1 + y * cos_theta1
    z_local = z
    L2 = 0.4412
    L3 = 0.39225
    D = math.sqrt(y_local ** 2 + z_local ** 2)
    C = (y_local ** 2 + z_local ** 2 + L3 ** 2 - L2 ** 2) / (2 * L3)
    C_over_D = C / D
    C_over_D = max(min(C_over_D, 1.0), -1.0)
    alpha = math.atan2(z_local, y_local)
    phi = alpha + math.acos(C_over_D)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    numerator_y = y_local - L3 * cos_phi
    numerator_z = z_local - L3 * sin_phi
    denominator = L2
    if denominator == 0:
        theta2 = 0.0
    else:
        cos_theta2 = numerator_y / denominator
        sin_theta2 = numerator_z / denominator
        theta2 = math.atan2(sin_theta2, cos_theta2)
    theta3 = phi - theta2
    return (theta1, theta2, theta3)