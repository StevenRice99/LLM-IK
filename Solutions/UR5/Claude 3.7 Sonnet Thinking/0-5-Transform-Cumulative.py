def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    L_tcp = 0.0823
    y1 = 0.13585
    y2 = -0.1197
    y3 = 0.093
    y_total = y1 + y2 + y3
    tcp_rz = 1.570796325
    px, py, pz = p
    roll, pitch, yaw = r

    def Rx(angle):
        c, s = (math.cos(angle), math.sin(angle))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(angle):
        c, s = (math.cos(angle), math.sin(angle))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(angle):
        c, s = (math.cos(angle), math.sin(angle))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_target = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R_tcp = Rz(tcp_rz)
    R_wrist = R_target @ R_tcp.T
    tcp_offset = np.array([0, L_tcp, 0])
    tcp_offset_world = R_target @ tcp_offset
    wc = np.array([px, py, pz]) - tcp_offset_world
    j6_offset = np.array([0, 0, L3])
    j5_pos = wc - R_wrist @ j6_offset
    candidates = []
    theta = math.atan2(j5_pos[1], j5_pos[0])
    r_xy = math.sqrt(j5_pos[0] ** 2 + j5_pos[1] ** 2)
    joint1_angles = []
    if r_xy < y_total:
        joint1_angles = [0, math.pi / 2, math.pi, -math.pi / 2, math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]
    else:
        try:
            offset = math.asin(y_total / r_xy)
            joint1_angles = [theta - offset, theta + math.pi - offset, theta + offset, theta - math.pi + offset]
        except ValueError:
            joint1_angles = [theta, theta + math.pi, theta + math.pi / 2, theta - math.pi / 2]
    for q1 in joint1_angles:
        while q1 > math.pi:
            q1 -= 2 * math.pi
        while q1 < -math.pi:
            q1 += 2 * math.pi
        R1 = Rz(q1)
        j5_in_base = j5_pos - np.array([0, y_total, 0])
        j5_in_j1 = R1.T @ j5_in_base
        R_1to_wrist = R1.T @ R_wrist
        phi = math.atan2(R_1to_wrist[0, 2], R_1to_wrist[2, 2])
        q5 = math.atan2(R_1to_wrist[1, 0], R_1to_wrist[1, 1])
        x, _, z = j5_in_j1
        r = math.sqrt(x ** 2 + z ** 2)
        arm_length = L1 + L2
        if r > arm_length + 0.001:
            continue
        cos_q3 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        for q3_sign in [-1, 1]:
            q3 = q3_sign * math.acos(cos_q3)
            k1 = L1 + L2 * math.cos(q3)
            k2 = L2 * math.sin(q3)
            gamma = math.atan2(k2, k1)
            q2 = math.atan2(x, z) - gamma
            q4 = phi - q2 - q3
            x_fk = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
            z_fk = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
            pos_error = math.sqrt((x_fk - x) ** 2 + (z_fk - z) ** 2)
            R2 = Ry(q2)
            R3 = Ry(q3)
            R4 = Ry(q4)
            R5 = Rz(q5)
            R_0to5 = R1 @ R2 @ R3 @ R4 @ R5
            R6_needed = R_0to5.T @ R_wrist
            q6 = math.atan2(R6_needed[0, 2], R6_needed[2, 2])
            R6 = Ry(q6)
            R_full = R_0to5 @ R6
            orient_error = np.linalg.norm(R_full - R_wrist, 'fro')
            total_error = pos_error + orient_error
            candidates.append((total_error, (q1, q2, q3, q4, q5, q6)))
    if not candidates:
        q1 = math.atan2(j5_pos[1], j5_pos[0])
        q2 = math.atan2(j5_pos[0], j5_pos[2])
        q3 = 0
        q4 = 0
        q5 = 0
        q6 = 0
        return (q1, q2, q3, q4, q5, q6)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]