import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, y)
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    x_1 = c1 * x + s1 * y
    y_1 = -s1 * x + c1 * y
    z_1 = z
    y_2 = y_1 - 0.13585
    l2 = math.sqrt(0.425 ** 2 + 0.1197 ** 2)
    l3 = 0.39225
    r = math.sqrt(x_1 ** 2 + y_2 ** 2 + z_1 ** 2)
    cos_theta3 = (r ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    r_proj = math.sqrt(x_1 ** 2 + z_1 ** 2)
    beta = math.atan2(z_1, x_1)
    alpha = math.atan2(l3 * math.sin(theta3), l2 + l3 * math.cos(theta3))
    theta2 = beta - alpha
    return (theta1, theta2, theta3)