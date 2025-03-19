def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    roll, pitch, yaw = r
    l1 = 0.425
    l2 = 0.39225
    d1 = 0.1197
    d2 = 0.093
    d3 = 0.09465

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    wrist_offset = R_target @ np.array([0, 0, d3])
    wrist_pos = np.array([x, y, z]) - wrist_offset
    wx, wy, wz = wrist_pos
    theta1 = np.arctan2(wx, wz)
    c1, s1 = (np.cos(theta1), np.sin(theta1))
    wx_1 = c1 * wx + s1 * wz
    wz_1 = -s1 * wx + c1 * wz
    wy_1 = wy
    wy_1_adj = wy_1 + d1
    D = np.sqrt(wx_1 ** 2 + wy_1_adj ** 2 + (wz_1 - l1) ** 2)
    cos_theta3 = (D ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    beta = np.arctan2(wy_1_adj, np.sqrt(wx_1 ** 2 + (wz_1 - l1) ** 2))
    cos_alpha = (l1 ** 2 + D ** 2 - l2 ** 2) / (2 * l1 * D)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    gamma = np.arctan2(wx_1, wz_1 - l1)
    theta2 = beta + alpha
    R_1 = rot_y(theta1)
    R_2 = rot_y(theta2)
    R_3 = rot_y(theta3)
    R_123 = R_1 @ R_2 @ R_3
    R_4_needed = np.transpose(R_123) @ R_target
    theta4 = np.arctan2(R_4_needed[1, 0], R_4_needed[0, 0])
    theta2 = np.arctan2(wy_1_adj, np.sqrt(wx_1 ** 2 + (wz_1 - l1) ** 2)) + np.arccos((l1 ** 2 + D ** 2 - l2 ** 2) / (2 * l1 * D))
    theta3 = np.arccos((l1 ** 2 + l2 ** 2 - D ** 2) / (2 * l1 * l2))
    theta3 = -theta3
    return (theta1, theta2, theta3, theta4)