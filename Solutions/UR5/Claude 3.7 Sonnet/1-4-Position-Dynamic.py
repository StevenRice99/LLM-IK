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
    y_offset = -0.1197
    theta4 = 0.0
    j4_x = x
    j4_y = y
    j4_z = z - L4_z
    j3_x = j4_x
    j3_y = j4_y - L3_y
    j3_z = j4_z
    r_xz = math.sqrt(j3_x ** 2 + j3_z ** 2)
    y_to_cover = j3_y - y_offset
    L2_effective = math.sqrt(L2 ** 2 + y_to_cover ** 2)
    phi = math.atan2(y_to_cover, L2)
    cos_alpha = (r_xz ** 2 + L1 ** 2 - L2_effective ** 2) / (2 * r_xz * L1)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta1 = math.atan2(j3_x, j3_z)
    if j3_x < 0 and j3_z < 0:
        theta1 -= alpha
    else:
        theta1 += alpha
    cos_beta = (L1 ** 2 + L2_effective ** 2 - r_xz ** 2) / (2 * L1 * L2_effective)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    beta = math.acos(cos_beta)
    if j3_y > y_offset:
        theta2 = math.pi - beta - phi
    else:
        theta2 = math.pi - beta + phi
    j3_calc_x = L1 * math.sin(theta1) + L2 * math.cos(phi) * math.sin(theta1 + theta2)
    j3_calc_z = L1 * math.cos(theta1) + L2 * math.cos(phi) * math.cos(theta1 + theta2)
    theta3 = math.atan2(j4_x - j3_calc_x, j4_z - j3_calc_z) - (theta1 + theta2)
    return (theta1, theta2, theta3, theta4)