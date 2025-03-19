def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    j2_offset_y = -0.1197
    j2_offset_z = 0.425
    j3_length = 0.39225
    j4_offset_y = 0.093
    j5_offset_z = 0.09465
    tcp_offset_y = 0.0823
    tcp_rz = 1.570796325

    def rpy_to_rot_matrix(rx, ry, rz):
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    R_target = rpy_to_rot_matrix(rx, ry, rz)
    R_tcp = rpy_to_rot_matrix(0, 0, tcp_rz)
    tcp_offset = np.array([0, tcp_offset_y, 0])
    wrist_pos = np.array([x, y, z]) - R_target @ tcp_offset
    R_wrist = R_target @ R_tcp.T
    theta1 = math.atan2(wrist_pos[0], wrist_pos[2])
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    R1 = np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]])
    wrist_in_j1 = R1.T @ wrist_pos
    j2_pos = np.array([0, j2_offset_y, j2_offset_z])
    v_j2_to_wrist = wrist_in_j1 - j2_pos
    v_yz = np.array([0, v_j2_to_wrist[1], v_j2_to_wrist[2]])
    L_yz = np.linalg.norm(v_yz)
    l2 = j3_length
    l3 = math.sqrt(j4_offset_y ** 2 + j5_offset_z ** 2)
    cos_theta3 = (L_yz ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    phi = math.atan2(v_yz[1], v_yz[2])
    cos_alpha = (l2 ** 2 + L_yz ** 2 - l3 ** 2) / (2 * l2 * L_yz)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta2 = phi - alpha
    c2, s2 = (math.cos(theta2), math.sin(theta2))
    c3, s3 = (math.cos(theta3), math.sin(theta3))
    R_0_to_3 = np.array([[c1 * c2 * c3 - c1 * s2 * s3, -c1 * c2 * s3 - c1 * s2 * c3, s1], [s1 * c2 * c3 - s1 * s2 * s3, -s1 * c2 * s3 - s1 * s2 * c3, -c1], [s2 * c3 + c2 * s3, -s2 * s3 + c2 * c3, 0]])
    R_3_to_5 = R_0_to_3.T @ R_wrist
    theta4 = math.atan2(R_3_to_5[1, 0], R_3_to_5[0, 0])
    c4, s4 = (math.cos(theta4), math.sin(theta4))
    R4 = np.array([[c4, -s4, 0], [s4, c4, 0], [0, 0, 1]])
    R_5 = R4.T @ R_3_to_5
    theta5 = math.atan2(-R_5[2, 0], R_5[2, 2])
    return (theta1, theta2, theta3, theta4, theta5)