import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    d_xz = math.sqrt(x ** 2 + z ** 2)
    y_target = y - 0.093
    z_target = d_xz - 0.425
    a = 0.39225
    b = 0.1197
    c = math.sqrt(y_target ** 2 + z_target ** 2)
    cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    cos_alpha = max(min(cos_alpha, 1), -1)
    alpha = math.acos(cos_alpha)
    cos_beta = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cos_beta = max(min(cos_beta, 1), -1)
    beta = math.acos(cos_beta)
    theta2 = math.atan2(y_target, z_target) - alpha
    theta3 = math.pi - beta
    return (theta1, theta2, theta3)