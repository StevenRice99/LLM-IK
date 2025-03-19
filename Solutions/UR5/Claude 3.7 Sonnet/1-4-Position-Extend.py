import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3_y = 0.093
    L4_z = 0.09465
    joint2_y = -0.1197
    theta4 = 0
    j4_x = x
    j4_y = y
    j4_z = z - L4_z
    j3_x = j4_x
    j3_y = j4_y + L3_y
    j3_z = j4_z
    r = math.sqrt(j3_x ** 2 + j3_z ** 2)
    theta1 = math.atan2(j3_x, j3_z)
    j3_y_rel = j3_y - joint2_y
    d = math.sqrt(j3_y_rel ** 2 + r ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_internal = math.acos(cos_theta2)
    alpha = math.atan2(j3_y_rel, r)
    beta = math.atan2(L2 * math.sin(theta2_internal), L1 + L2 * math.cos(theta2_internal))
    cross_product = r * (L1 + L2 * math.cos(theta2_internal)) - j3_y_rel * (L2 * math.sin(theta2_internal))
    if cross_product < 0:
        theta2_internal = -theta2_internal
        beta = -beta
    theta2 = alpha - beta
    theta3 = -theta2_internal
    return (theta1, theta2, theta3, theta4)