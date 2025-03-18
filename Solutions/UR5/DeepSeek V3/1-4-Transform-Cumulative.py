import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    theta1 = math.atan2(p[0], p[2])
    x2 = p[0] - 0.1197 * math.sin(theta1)
    z2 = p[2] - 0.1197 * math.cos(theta1)
    d = math.sqrt(x2 ** 2 + z2 ** 2)
    a = 0.425
    b = 0.39225
    cos_theta2 = (a ** 2 + d ** 2 - b ** 2) / (2 * a * d)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.atan2(math.sqrt(1 - cos_theta2 ** 2), cos_theta2)
    x3 = x2 - 0.39225 * math.sin(theta2)
    z3 = z2 - 0.39225 * math.cos(theta2)
    d = math.sqrt(x3 ** 2 + z3 ** 2)
    a = 0.39225
    b = 0.093
    cos_theta3 = (a ** 2 + d ** 2 - b ** 2) / (2 * a * d)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.atan2(math.sqrt(1 - cos_theta3 ** 2), cos_theta3)
    theta4 = math.atan2(r[1], r[0])
    return (theta1, theta2, theta3, theta4)