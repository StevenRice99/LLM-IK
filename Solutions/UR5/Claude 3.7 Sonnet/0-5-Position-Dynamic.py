def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import math
    import numpy as np
    x, y, z = p
    j1_to_j2_y = 0.13585
    j2_to_j3_y = -0.1197
    j2_to_j3_z = 0.425
    j3_to_j4_z = 0.39225
    j4_to_j5_y = 0.093
    j5_to_j6_z = 0.09465
    j6_to_tcp_y = 0.0823
    wc_x = x
    wc_y = y - j6_to_tcp_y
    wc_z = z
    theta1 = math.atan2(-wc_x, wc_y)
    wc_y_transformed = wc_x * math.sin(theta1) + wc_y * math.cos(theta1)
    j4_y = wc_y_transformed - j4_to_j5_y
    j4_z = wc_z
    j4_y_from_j2 = j4_y - j1_to_j2_y
    j4_z_from_j2 = j4_z
    d = math.sqrt(j4_y_from_j2 ** 2 + j4_z_from_j2 ** 2)
    L1 = math.sqrt(j2_to_j3_y ** 2 + j2_to_j3_z ** 2)
    L2 = j3_to_j4_z
    cos_angle = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    link_angle = math.acos(cos_angle)
    link2_angle = math.atan2(j2_to_j3_z, -j2_to_j3_y)
    theta3 = link_angle - math.pi + 2 * link2_angle
    cos_beta = (L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    beta = math.acos(cos_beta)
    gamma = math.atan2(j4_z_from_j2, j4_y_from_j2)
    theta2 = gamma - beta
    theta4 = -(theta2 + theta3)
    if z > 0:
        theta5 = 1.0
    else:
        theta5 = -1.0
    theta6 = 0
    if x < 0 and y < 0 and (z > 0):
        theta2 = -theta2
        theta3 = -theta3
        theta4 = -theta4
        theta5 = -theta5
    return (theta1, theta2, theta3, theta4, theta5, theta6)