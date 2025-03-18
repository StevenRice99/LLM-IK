import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_d, y_d, z_d = p
    z_adjusted = z_d - 0.39225
    theta1 = math.atan2(y_d, x_d)
    r = math.sqrt(x_d ** 2 + y_d ** 2)
    d = math.sqrt(r ** 2 + z_adjusted ** 2)
    cos_theta2 = (0.425 ** 2 + d ** 2 - 0.1197 ** 2) / (2 * 0.425 * d)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    cos_theta3 = (0.425 ** 2 + 0.1197 ** 2 - d ** 2) / (2 * 0.425 * 0.1197)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    return (theta1, theta2, theta3)