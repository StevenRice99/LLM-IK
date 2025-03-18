import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    y_target, z_target, _ = p
    L1 = 0.39225
    L2 = 0.093
    r = math.sqrt(y_target ** 2 + z_target ** 2)
    cos_theta2 = (L1 ** 2 + L2 ** 2 - r ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    A = L1 + L2 * cos_theta2
    B = L2 * math.sin(theta2)
    denominator = A ** 2 + B ** 2
    if denominator == 0:
        sin_theta1 = 0.0
        cos_theta1 = 1.0
    else:
        sin_theta1 = (A * y_target - B * z_target) / denominator
        cos_theta1 = (A * z_target + B * y_target) / denominator
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)