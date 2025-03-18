import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x_d, y_d, z_d = p
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    r = math.sqrt(x_d ** 2 + y_d ** 2)
    theta1 = math.atan2(y_d, x_d)
    z_adjusted = z_d - l3
    d = math.sqrt(r ** 2 + z_adjusted ** 2)
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    alpha = math.atan2(z_adjusted, r)
    beta = math.asin(l2 * math.sin(theta2) / d)
    theta3 = alpha - beta
    return (theta1, theta2, theta3)