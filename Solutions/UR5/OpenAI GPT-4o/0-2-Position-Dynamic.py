import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    d2 = 0.425
    d3 = 0.39225
    theta1 = math.atan2(-x, y)
    r = math.sqrt(x ** 2 + y ** 2)
    z2 = z - d3
    L = math.sqrt(r ** 2 + (z2 - d1) ** 2)
    cos_theta3 = (d2 ** 2 + d3 ** 2 - L ** 2) / (2 * d2 * d3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    cos_theta2 = (d2 ** 2 + L ** 2 - d3 ** 2) / (2 * d2 * L)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    if z2 < d1:
        theta2 = -theta2
    return (theta1, theta2, theta3)