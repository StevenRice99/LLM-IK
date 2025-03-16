import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_target, y_target, z_target = p
    l1 = 0.425
    l2 = 0.39225
    d_tcp = 0.093
    y_target_adjusted = y_target + d_tcp
    theta1 = math.atan2(x_target, z_target)
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    d = math.sqrt(r ** 2 + y_target_adjusted ** 2)
    cos_angle = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    theta2 = math.atan2(y_target_adjusted, r) - math.acos(cos_angle)
    theta3 = 0.0
    return (theta1, theta2, theta3)