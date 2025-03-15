import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_target, y_target, z_target = p
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.093
    theta1 = math.atan2(-x_target, y_target)
    y_adjusted = y_target - d3
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta3 = (r ** 2 - d1 ** 2 - d2 ** 2) / (2 * d1 * d2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    phi = math.atan2(x_target, z_target)
    beta = math.atan2(d2 * sin_theta3, d1 + d2 * cos_theta3)
    theta2 = phi - beta
    theta4 = -theta2 - theta3
    return (theta1, theta2, theta3, theta4)