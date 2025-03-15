import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.1197
    L2 = 0.39225
    z_offset = 0.425
    r = math.sqrt(x ** 2 + y ** 2 + (z - z_offset) ** 2)
    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(z - z_offset, math.sqrt(x ** 2 + y ** 2))
    psi = math.asin(L2 * math.sin(theta2) / r)
    theta1 = phi - psi
    return (theta1, theta2)