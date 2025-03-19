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

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    d2 = -0.1197
    d3 = 0.425
    d5 = 0.39225
    d6 = 0.093
    d8 = 0.09465
    d9 = 0.0823
    tcp_rot_offset = rot_z(1.570796325)
    R_base_to_tcp = R_target
    tcp_offset_world = R_base_to_tcp @ np.array([0, d9, 0])
    wrist_center = np.array([x, y, z]) - tcp_offset_world
    theta1 = math.atan2(wrist_center[0], wrist_center[2])
    R1 = rot_y(theta1)
    wrist_in_frame1 = R1.T @ wrist_center
    joint2_pos = np.array([0, d2, d3])
    wrist_rel_joint2 = wrist_in_frame1 - joint2_pos
    dist_2_to_wrist = np.linalg.norm(wrist_rel_joint2)
    l3_to_wrist = math.sqrt(d6 ** 2 + d8 ** 2)
    phi = math.atan2(d6, d8)
    cos_theta3 = (d5 ** 2 + l3_to_wrist ** 2 - dist_2_to_wrist ** 2) / (2 * d5 * l3_to_wrist)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    angle_3_to_wrist = math.acos(cos_theta3)
    theta3 = math.pi - angle_3_to_wrist - phi
    beta = math.atan2(wrist_rel_joint2[1], wrist_rel_joint2[2])
    cos_alpha = (d5 ** 2 + dist_2_to_wrist ** 2 - l3_to_wrist ** 2) / (2 * d5 * dist_2_to_wrist)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta2 = beta - alpha
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R_0_to_3 = R1 @ R2 @ R3
    R_3_to_tcp = R_0_to_3.T @ R_target
    theta4 = math.atan2(R_3_to_tcp[1, 0], R_3_to_tcp[0, 0])
    R4 = rot_z(theta4)
    R_4_to_tcp = R4.T @ R_3_to_tcp
    theta5 = math.atan2(-R_4_to_tcp[0, 2], R_4_to_tcp[2, 2])
    if abs(theta1) > math.pi:
        if theta1 > 0:
            theta1 -= 2 * math.pi
        else:
            theta1 += 2 * math.pi
    if abs(theta2) > math.pi / 2:
        if theta2 > 0:
            theta2 = math.pi - theta2
            theta3 = -theta3
            theta4 = theta4 + math.pi
            if theta4 > math.pi:
                theta4 -= 2 * math.pi
        else:
            theta2 = -math.pi - theta2
            theta3 = -theta3
            theta4 = theta4 + math.pi
            if theta4 > math.pi:
                theta4 -= 2 * math.pi
    if abs(theta5) > math.pi:
        if theta5 > 0:
            theta5 -= 2 * math.pi
        else:
            theta5 += 2 * math.pi
    return (theta1, theta2, theta3, theta4, theta5)