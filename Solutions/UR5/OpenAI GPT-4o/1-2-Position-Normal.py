import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    y -= -0.1197
    r = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(z, x)
    psi = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = phi - psi
    return (theta1, theta2)