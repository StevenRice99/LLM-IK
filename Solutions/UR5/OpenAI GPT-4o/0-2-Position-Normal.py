import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    d2 = 0.425
    d3 = 0.39225
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    z_eff = z - d3
    y_eff = r - d1
    D = (y_eff ** 2 + z_eff ** 2 - d2 ** 2 - d3 ** 2) / (2 * d2 * d3)
    D = max(-1, min(1, D))
    theta3 = math.atan2(math.sqrt(1 - D ** 2), D)
    phi2 = math.atan2(z_eff, y_eff)
    phi1 = math.atan2(d3 * math.sin(theta3), d2 + d3 * math.cos(theta3))
    theta2 = phi2 - phi1
    return (theta1, theta2, theta3)