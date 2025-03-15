import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    z -= 0.39225
    theta1 = math.atan2(-x, y)
    y_local = math.sqrt(x ** 2 + y ** 2) - 0.13585
    d1 = 0.425
    d2 = 0.39225
    r = math.sqrt(y_local ** 2 + z ** 2)
    cos_theta3 = (r ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    phi = math.atan2(z, y_local)
    beta = math.atan2(d2 * sin_theta3, d1 + d2 * cos_theta3)
    theta2 = phi - beta
    return (theta1, theta2, theta3)