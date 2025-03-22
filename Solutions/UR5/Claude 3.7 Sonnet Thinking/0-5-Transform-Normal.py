def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    d1 = 0
    a1 = 0
    a2 = 0
    d2 = 0.13585
    a3 = 0
    d3 = 0
    a4 = 0
    d4 = 0.39225
    a5 = 0
    d5 = 0.093
    a6 = 0
    d6 = 0.09465
    d7 = 0.0823
    z_offset_2_3 = 0.425
    y_offset_2_3 = -0.1197

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    def euler_to_rotation(rx, ry, rz):
        R_x = rot_x(rx)
        R_y = rot_y(ry)
        R_z = rot_z(rz)
        return np.dot(R_z, np.dot(R_y, R_x))
    target_pos = np.array(p)
    target_rot = euler_to_rotation(r[0], r[1], r[2])
    tcp_rot_offset = rot_z(1.570796325)
    tool_offset = np.array([0, d7, 0])
    wrist_offset = np.array([0, 0, d6])
    wrist_rot = np.dot(target_rot, np.linalg.inv(tcp_rot_offset))
    wc = target_pos - np.dot(wrist_rot, tool_offset) - np.dot(wrist_rot, wrist_offset)
    theta1 = np.arctan2(wc[1], wc[0])
    if np.abs(wc[0]) < 1e-10 and np.abs(wc[1]) < 1e-10:
        theta1 = 0
    R0_1 = rot_z(theta1)
    wc_in_1 = np.dot(R0_1.T, wc)
    wc_from_2 = wc_in_1.copy()
    wc_from_2[1] -= d2
    r_2_wc = np.sqrt(wc_from_2[1] ** 2 + wc_from_2[2] ** 2)
    a2_3 = np.sqrt(y_offset_2_3 ** 2 + z_offset_2_3 ** 2)
    a3_wc = d4
    gamma = np.arctan2(z_offset_2_3, y_offset_2_3)
    cos_alpha = (r_2_wc ** 2 + a2_3 ** 2 - a3_wc ** 2) / (2 * r_2_wc * a2_3)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    beta = np.arctan2(wc_from_2[2], wc_from_2[1])
    theta2 = -(beta - alpha + gamma)
    cos_theta3 = (a2_3 ** 2 + a3_wc ** 2 - r_2_wc ** 2) / (2 * a2_3 * a3_wc)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3) - np.pi / 2 - gamma
    R0_1 = rot_z(theta1)
    R1_2 = rot_y(theta2)
    R2_3 = rot_y(theta3)
    R0_3 = np.dot(R0_1, np.dot(R1_2, R2_3))
    R3_6 = np.dot(R0_3.T, wrist_rot)
    singularity_threshold = 1e-10
    if np.abs(R3_6[1, 0]) > 1.0 - singularity_threshold and np.abs(R3_6[1, 1]) < singularity_threshold:
        theta4 = 0.0
        theta5 = np.pi / 2 if R3_6[1, 0] > 0 else -np.pi / 2
        theta6 = np.arctan2(R3_6[0, 1], -R3_6[2, 1])
    else:
        theta5 = np.arctan2(R3_6[1, 0], R3_6[1, 1])
        sin_theta5 = np.sin(theta5)
        cos_theta5 = np.cos(theta5)
        theta4 = np.arctan2(-R3_6[0, 2], R3_6[2, 2])
        theta6 = np.arctan2(-R3_6[1, 2], R3_6[1, 1] * cos_theta5 + R3_6[1, 0] * sin_theta5)
    return (theta1, theta2, theta3, theta4, theta5, theta6)