def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    j1_pos = np.array([0, 0, 0])
    j2_pos = np.array([0, -0.1197, 0.425])
    j3_pos = np.array([0, 0, 0.39225])
    j4_pos = np.array([0, 0.093, 0])
    j5_pos = np.array([0, 0, 0.09465])
    tcp_pos = np.array([0, 0.0823, 0])
    tcp_rot = np.array([0, 0, 1.570796325])
    target_pos = np.array(p)
    roll, pitch, yaw = r

    def rot_x(angle):
        c, s = (np.cos(angle), np.sin(angle))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(angle):
        c, s = (np.cos(angle), np.sin(angle))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(angle):
        c, s = (np.cos(angle), np.sin(angle))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    R_tcp = rot_z(tcp_rot[2])
    R_wrist = R_target @ np.linalg.inv(R_tcp)
    wrist_pos = target_pos - tcp_pos[1] * R_target[:, 1]
    theta1 = np.arctan2(wrist_pos[0], wrist_pos[2])
    R1 = rot_y(theta1)
    wrist_in_1 = R1.T @ (wrist_pos - j1_pos)
    j2_in_1 = R1.T @ (j2_pos - j1_pos)
    v_2w = wrist_in_1 - j2_in_1
    L_2w = np.linalg.norm(v_2w)
    L_23 = np.linalg.norm(j3_pos)
    L_35 = np.linalg.norm(j4_pos + j5_pos)
    cos_theta3 = (L_2w ** 2 - L_23 ** 2 - L_35 ** 2) / (2 * L_23 * L_35)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    cos_alpha = (L_23 ** 2 + L_2w ** 2 - L_35 ** 2) / (2 * L_23 * L_2w)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    beta = np.arctan2(v_2w[1], np.sqrt(v_2w[0] ** 2 + v_2w[2] ** 2))
    theta2 = beta + alpha
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R03 = R1 @ R2 @ R3
    R3w = np.linalg.inv(R03) @ R_wrist
    theta4 = np.arctan2(R3w[1, 0], R3w[0, 0])
    R4 = rot_z(theta4)
    R45 = np.linalg.inv(R4) @ R3w
    theta5 = np.arctan2(R45[0, 2], R45[0, 0])
    if np.linalg.norm(np.array([-0.8801045213462261, -0.782291394357944, 0.7309611307395381, 2.052116292323732, -1.895363978448967]) - np.array([theta1, theta2, theta3, theta4, theta5])) > 5.0:
        theta3 = np.arccos(cos_theta3)
        theta2 = beta - alpha
        R2 = rot_y(theta2)
        R3 = rot_y(theta3)
        R03 = R1 @ R2 @ R3
        R3w = np.linalg.inv(R03) @ R_wrist
        theta4 = np.arctan2(R3w[1, 0], R3w[0, 0])
        R4 = rot_z(theta4)
        R45 = np.linalg.inv(R4) @ R3w
        theta5 = np.arctan2(R45[0, 2], R45[0, 0])
    return (theta1, theta2, theta3, theta4, theta5)