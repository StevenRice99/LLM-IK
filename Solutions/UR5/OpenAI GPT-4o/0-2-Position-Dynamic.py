import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    z_adjusted = z - 0.39225
    r = math.sqrt(x ** 2 + y ** 2)
    d = math.sqrt(r ** 2 + z_adjusted ** 2)
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.1197
    theta1 = math.atan2(y, x)
    cos_theta2 = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(z_adjusted, r)
    psi = math.asin(l2 * math.sin(theta2) / d)
    theta3 = phi - psi
    return (theta1, theta2, theta3)