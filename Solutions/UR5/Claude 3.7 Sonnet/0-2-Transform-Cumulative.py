import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, y)
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    x_1 = c1 * x + s1 * y
    y_1 = -s1 * x + c1 * y
    z_1 = z
    l1_y = 0.13585
    l2_z = 0.425
    l2_y = -0.1197
    l3_z = 0.39225
    y_2 = y_1 - l1_y
    x_2 = x_1
    z_2 = z_1
    r_2 = math.sqrt(x_2 ** 2 + z_2 ** 2)
    l2 = math.sqrt(l2_z ** 2 + l2_y ** 2)
    l3 = l3_z
    alpha2 = math.atan2(l2_y, l2_z)
    cos_theta3 = (r_2 ** 2 + y_2 ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(y_2, r_2)
    gamma = math.atan2(l3 * math.sin(theta3), l2 + l3 * math.cos(theta3))
    theta2 = beta - gamma - alpha2
    return (theta1, theta2, theta3)