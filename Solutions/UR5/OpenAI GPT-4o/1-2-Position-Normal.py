import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.425
    d2 = 0.39225
    offset_y = -0.1197
    y_adjusted = y - offset_y
    z_eff = z - d2
    r = math.sqrt(y_adjusted ** 2 + z_eff ** 2)
    cos_theta2 = (r ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(z_eff, y_adjusted)
    psi = math.atan2(d2 * math.sin(theta2), d1 + d2 * math.cos(theta2))
    theta1 = phi - psi
    theta1 = math.atan2(math.sin(theta1), math.cos(theta1))
    theta2 = math.atan2(math.sin(theta2), math.cos(theta2))
    return (theta1, theta2)