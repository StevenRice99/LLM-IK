import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    r = math.sqrt(y ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    phi = math.atan2(z, y)
    psi = math.atan2(L2 * sin_theta2, L1 + L2 * cos_theta2)
    theta1 = phi - psi
    theta1 = math.atan2(z, y) - math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    return (theta1, theta2)