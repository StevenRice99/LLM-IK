import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta_1 = math.atan2(-x, y)
    d3 = 0.093
    y_adjusted = y - d3
    d1 = 0.425
    d2 = 0.39225
    r = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta_2 = math.atan2(sin_theta2, cos_theta2)
    phi = math.atan2(x, z)
    beta = math.atan2(d2 * sin_theta2, d1 + d2 * cos_theta2)
    theta_3 = phi - beta
    theta_4 = math.atan2(y_adjusted, r) - theta_2
    return (theta_1, theta_2, theta_3, theta_4)