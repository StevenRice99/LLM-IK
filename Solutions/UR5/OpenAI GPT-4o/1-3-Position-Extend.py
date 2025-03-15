import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.093
    y_adjusted = y_target + d3
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta2 = (r ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    phi = math.atan2(x_target, z_target)
    beta = math.atan2(d2 * sin_theta2, d1 + d2 * cos_theta2)
    theta1 = phi - beta
    theta3 = math.atan2(y_adjusted, r) - theta2
    return (theta1, theta2, theta3)