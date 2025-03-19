import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    j1_to_j2_z = 0.13585
    j2_to_j3_y = -0.1197
    j2_to_j3_z = 0.425
    j3_to_j4_z = 0.39225
    j4_to_j5_y = 0.093
    j5_to_tcp_z = 0.09465
    theta1 = math.atan2(x, y)
    r = math.sqrt(x ** 2 + y ** 2)
    wc_z = z - j5_to_tcp_z
    wc_r = r - j4_to_j5_y
    wc_x = wc_r * math.sin(theta1)
    wc_y = wc_r * math.cos(theta1)
    wc_rel_z = wc_z - j1_to_j2_z
    d = math.sqrt(wc_r ** 2 + wc_rel_z ** 2)
    L1 = math.sqrt(j2_to_j3_z ** 2 + j2_to_j3_y ** 2)
    L2 = j3_to_j4_z
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    if z < 0:
        theta3 = -math.acos(cos_theta3)
    else:
        theta3 = math.acos(cos_theta3)
    gamma = math.atan2(wc_rel_z, wc_r)
    alpha = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = gamma - alpha
    if z >= 0:
        theta4 = -(theta2 + theta3)
    else:
        theta4 = -(theta2 + theta3)
    theta5 = 0.0
    if z < 0 and theta3 > 0 or (z >= 0 and wc_r < 0.2 and (wc_z < 0.3)):
        theta1 = math.atan2(-x, -y)
        theta2 = -theta2
        theta3 = -theta3
        theta4 = -theta4
    return (theta1, theta2, theta3, theta4, theta5)