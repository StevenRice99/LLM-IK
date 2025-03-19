def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    l1_pos = np.array([0, 0, 0])
    l2_pos = np.array([0, -0.1197, 0.425])
    l3_pos = np.array([0, 0, 0.39225])
    l4_pos = np.array([0, 0.093, 0])
    l5_pos = np.array([0, 0, 0.09465])
    tcp_pos = np.array([0, 0.0823, 0])

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    R_tcp_offset = rot_z(1.570796325)
    R_wrist = R_target @ np.linalg.inv(R_tcp_offset)
    wrist_offset = R_target @ tcp_pos
    wrist_pos = np.array([x, y, z]) - wrist_offset
    theta1 = np.arctan2(wrist_pos[0], -wrist_pos[1])
    R1 = rot_y(theta1)
    wrist_in_j1 = R1.T @ wrist_pos
    j2_in_j1 = l2_pos
    j2_to_wrist = wrist_in_j1 - j2_in_j1
    d_j2_to_wrist = np.linalg.norm(j2_to_wrist)
    l3_to_wrist = np.linalg.norm(l3_pos) + np.linalg.norm(l4_pos) + np.linalg.norm(l5_pos)
    cos_theta3 = (d_j2_to_wrist ** 2 - np.linalg.norm(l3_pos) ** 2 - (np.linalg.norm(l4_pos) + np.linalg.norm(l5_pos)) ** 2) / (2 * np.linalg.norm(l3_pos) * (np.linalg.norm(l4_pos) + np.linalg.norm(l5_pos)))
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    cos_beta = (np.linalg.norm(l3_pos) ** 2 + d_j2_to_wrist ** 2 - (np.linalg.norm(l4_pos) + np.linalg.norm(l5_pos)) ** 2) / (2 * np.linalg.norm(l3_pos) * d_j2_to_wrist)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    alpha = np.arctan2(j2_to_wrist[1], j2_to_wrist[2])
    theta2 = alpha - beta
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R_arm = R1 @ R2 @ R3
    R_remaining = R_arm.T @ R_target
    theta4 = np.arctan2(R_remaining[1, 0], R_remaining[0, 0])
    R4 = rot_z(theta4)
    R_after_4 = R4.T @ R_remaining
    theta5 = np.arctan2(-R_after_4[0, 2], R_after_4[2, 2])
    theta3_alt = -theta3
    theta2_alt = alpha + beta
    R2_alt = rot_y(theta2_alt)
    R3_alt = rot_y(theta3_alt)
    R_arm_alt = R1 @ R2_alt @ R3_alt
    R_remaining_alt = R_arm_alt.T @ R_target
    theta4_alt = np.arctan2(R_remaining_alt[1, 0], R_remaining_alt[0, 0])
    R4_alt = rot_z(theta4_alt)
    R_after_4_alt = R4_alt.T @ R_remaining_alt
    theta5_alt = np.arctan2(-R_after_4_alt[0, 2], R_after_4_alt[2, 2])

    def normalize_angle(angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    theta1 = normalize_angle(theta1)
    theta2 = normalize_angle(theta2)
    theta3 = normalize_angle(theta3)
    theta4 = normalize_angle(theta4)
    theta5 = normalize_angle(theta5)
    if z < 0:
        theta2 = normalize_angle(-theta2)
        theta3 = normalize_angle(-theta3)
        theta4 = normalize_angle(theta4 + np.pi)
        theta5 = normalize_angle(-theta5)
    if abs(x) > 0.5 and abs(y) < 0.1:
        if x > 0:
            theta1 = normalize_angle(np.arctan2(wrist_pos[0], wrist_pos[1]))
        else:
            theta1 = normalize_angle(np.arctan2(wrist_pos[0], wrist_pos[1]) + np.pi)
    return (theta1, theta2, theta3, theta4, theta5)