import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    a2 = 0.425
    a3 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    d7 = 0.0823
    wc_x = x
    wc_y = y - d7
    wc_z = z - d6
    theta1 = math.atan2(wc_x, wc_y)
    r = math.sqrt(wc_x ** 2 + wc_y ** 2) - d5
    s = wc_z - d1
    D = math.sqrt(r ** 2 + s ** 2)
    cos_theta3 = (D ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    alpha = math.atan2(s, r)
    beta = math.atan2(a3 * math.sin(abs(theta3)), a2 + a3 * math.cos(theta3))
    theta2 = alpha - beta
    theta4 = -(theta2 + theta3)
    if z < 0:
        theta5 = -math.pi / 2 if x > 0 else math.pi / 2
    else:
        theta5 = math.pi / 2 if y < 0 else -math.pi / 2
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)