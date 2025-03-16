import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    l1 = 0.425
    l2 = 0.39225
    y_offset1 = -0.1197
    y_offset_tcp = 0.093
    theta1 = math.atan2(x_target, z_target)
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    y_adjusted = y_target - y_offset_tcp
    d = math.sqrt(r ** 2 + (y_adjusted - y_offset1) ** 2)
    cos_theta3 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(r, y_adjusted - y_offset1)
    alpha = math.atan2(l2 * math.sin(theta3), l1 + l2 * math.cos(theta3))
    theta2 = beta - alpha
    return (theta1, theta2, theta3)