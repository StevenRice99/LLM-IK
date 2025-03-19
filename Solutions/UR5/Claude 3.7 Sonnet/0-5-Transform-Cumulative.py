def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    px, py, pz = p
    rx, ry, rz = r
    L1_pos = np.array([0, 0, 0])
    L2_pos = np.array([0, 0.13585, 0])
    L3_pos = np.array([0, -0.1197, 0.425])
    L4_pos = np.array([0, 0, 0.39225])
    L5_pos = np.array([0, 0.093, 0])
    L6_pos = np.array([0, 0, 0.09465])
    TCP_pos = np.array([0, 0.0823, 0])
    TCP_ori = np.array([0, 0, 1.570796325])

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    R_tcp_offset = rot_z(TCP_ori[2])
    R_wrist = R_target @ R_tcp_offset.T
    tcp_offset_world = R_target @ TCP_pos
    wrist_pos = np.array([px, py, pz]) - tcp_offset_world
    theta1 = math.atan2(wrist_pos[0], wrist_pos[1])
    R1 = rot_z(theta1)
    wrist_in_1 = R1.T @ wrist_pos
    joint2_pos = L2_pos
    v = wrist_in_1 - joint2_pos
    d = np.linalg.norm(v)
    a2 = math.sqrt(L3_pos[1] ** 2 + L3_pos[2] ** 2)
    a3 = L4_pos[2]
    cos_theta3 = (d ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    v_yz = np.array([0, v[1], v[2]])
    phi = math.atan2(v[1], v[2])
    psi = math.atan2(a3 * math.sin(-theta3), a2 + a3 * math.cos(-theta3))
    theta2 = phi - psi
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R_0_3 = R1 @ R2 @ R3
    R_3_6 = R_0_3.T @ R_wrist
    theta5 = math.atan2(math.sqrt(R_3_6[1, 0] ** 2 + R_3_6[1, 2] ** 2), R_3_6[1, 1])
    if abs(math.sin(theta5)) < 1e-10:
        theta4 = 0
        theta6 = math.atan2(R_3_6[0, 2], R_3_6[2, 2])
    else:
        theta4 = math.atan2(R_3_6[1, 0], -R_3_6[1, 2])
        theta6 = math.atan2(R_3_6[0, 1], R_3_6[2, 1])
    return (theta1, theta2, theta3, theta4, theta5, theta6)