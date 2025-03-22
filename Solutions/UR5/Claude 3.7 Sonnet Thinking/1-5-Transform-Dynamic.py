import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    d1_y = -0.1197
    d1_z = 0.425
    d2_z = 0.39225
    d3_y = 0.093
    d4_z = 0.09465
    d_tcp_y = 0.0823
    tcp_rot_z = 1.570796325
    target_pos = np.array(p)
    roll, pitch, yaw = r

    def rot_x(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    tcp_offset = np.array([0, d_tcp_y, 0])
    wrist_center = target_pos - R_target @ tcp_offset
    theta1 = np.arctan2(wrist_center[0], wrist_center[2])
    R1 = rot_y(theta1)
    wrist_in_j1 = R1.T @ wrist_center
    j2_pos = np.array([0, d1_y, d1_z])
    v_j2_to_wrist = wrist_in_j1 - j2_pos
    d_j2_to_wrist = np.linalg.norm(v_j2_to_wrist)
    l2 = d2_z
    l3 = np.sqrt(d3_y ** 2 + d4_z ** 2)
    phi = np.arctan2(d3_y, d4_z)
    cos_alpha = (l2 ** 2 + d_j2_to_wrist ** 2 - l3 ** 2) / (2 * l2 * d_j2_to_wrist)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    beta = np.arctan2(v_j2_to_wrist[1], v_j2_to_wrist[2])
    theta2 = beta - alpha
    cos_gamma = (l2 ** 2 + l3 ** 2 - d_j2_to_wrist ** 2) / (2 * l2 * l3)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    theta3 = np.pi - gamma - phi
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R_0_to_3 = R1 @ R2 @ R3
    R_desired = R_target @ rot_z(-tcp_rot_z)
    R_3_to_desired = R_0_to_3.T @ R_desired
    sin_theta5 = -R_3_to_desired[2, 0]
    cos_theta5 = R_3_to_desired[2, 2]
    theta5 = np.arctan2(sin_theta5, cos_theta5)
    if abs(cos_theta5) > 1e-06:
        cos_theta4 = R_3_to_desired[0, 0] / cos_theta5
        sin_theta4 = R_3_to_desired[1, 0] / cos_theta5
    else:
        cos_theta4 = R_3_to_desired[1, 1]
        sin_theta4 = -R_3_to_desired[0, 1]
    theta4 = np.arctan2(sin_theta4, cos_theta4)
    return (theta1, theta2, theta3, theta4, theta5)