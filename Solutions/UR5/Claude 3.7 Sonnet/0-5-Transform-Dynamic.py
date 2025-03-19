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
    roll, pitch, yaw = r

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def rot_y(angle):
        return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def rot_z(angle):
        return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    d1 = 0.13585
    a2 = 0.425
    d2 = -0.1197
    a3 = 0.39225
    d4 = 0.093
    d5 = 0.09465
    d6 = 0.0823
    tcp_rot_z = 1.570796325
    R_tcp = rot_z(tcp_rot_z)
    R_wrist = R_target @ R_tcp.T
    wrist_vector = R_target @ np.array([0, d6, 0])
    wrist_pos = np.array([px, py, pz]) - wrist_vector
    j5_offset = R_wrist @ np.array([0, 0, d5])
    j5_pos = wrist_pos - j5_offset
    theta1 = math.atan2(j5_pos[1], j5_pos[0])
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    j5_in_j1 = np.array([c1 * j5_pos[0] + s1 * j5_pos[1], -s1 * j5_pos[0] + c1 * j5_pos[1], j5_pos[2]])
    j2_in_j1 = np.array([0, 0, d1])
    v2_5 = j5_in_j1 - j2_in_j1
    d2_5 = np.linalg.norm(v2_5)
    l2_3 = math.sqrt(a2 ** 2 + d2 ** 2)
    l3_5 = math.sqrt(a3 ** 2 + d4 ** 2)
    cos_theta3 = (d2_5 ** 2 - l2_3 ** 2 - l3_5 ** 2) / (2 * l2_3 * l3_5)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    phi = math.atan2(v2_5[2], math.sqrt(v2_5[0] ** 2 + v2_5[1] ** 2))
    cos_alpha = (l2_3 ** 2 + d2_5 ** 2 - l3_5 ** 2) / (2 * l2_3 * d2_5)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta2 = phi - alpha
    R1 = rot_z(theta1)
    R2 = rot_y(theta2)
    R3 = rot_y(theta3)
    R0_3 = R1 @ R2 @ R3
    R3_6 = R0_3.T @ R_wrist
    if abs(R3_6[1, 2]) > 0.9999:
        theta5 = math.pi / 2 * np.sign(R3_6[1, 2])
        theta4 = 0
        theta6 = math.atan2(R3_6[0, 1], R3_6[0, 0])
    else:
        theta5 = math.atan2(math.sqrt(R3_6[0, 2] ** 2 + R3_6[2, 2] ** 2), R3_6[1, 2])
        theta4 = math.atan2(R3_6[0, 2], -R3_6[2, 2])
        theta6 = math.atan2(R3_6[1, 0], R3_6[1, 1])
    return (theta1, theta2, theta3, theta4, theta5, theta6)