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
    Y2 = 0.13585
    Y3 = -0.1197
    Y5 = 0.093
    TCP_Y = 0.0823
    Y_OFFSET = Y2 + Y3 + Y5

    def Rx(angle):
        return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def Ry(angle):
        return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def Rz(angle):
        return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    def normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    px, py, pz = p
    roll, pitch, yaw = r
    R_target = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R_tcp_offset = Rz(math.pi / 2)
    R_wrist = R_target @ R_tcp_offset.T
    tcp_offset_world = R_target @ np.array([0, TCP_Y, 0])
    wrist_pos = np.array([px, py, pz]) - tcp_offset_world
    wx, wy, wz = wrist_pos
    xy_dist = math.sqrt(wx ** 2 + wy ** 2)
    if xy_dist < 1e-06:
        q1 = 0.0
    else:
        wrist_angle = math.atan2(wy, wx)
        if xy_dist < Y_OFFSET:
            q1 = wrist_angle
        else:
            ratio = Y_OFFSET / xy_dist
            ratio = max(-1.0, min(1.0, ratio))
            offset_angle = math.asin(ratio)
            q1_sol1 = wrist_angle - offset_angle
            q1_sol2 = math.pi - wrist_angle + offset_angle

            def q1_error(q1_val):
                Rz_inv = Rz(-q1_val)
                M = Rz_inv @ R_wrist
                return abs(M[1, 2])
            err1 = q1_error(q1_sol1)
            err2 = q1_error(q1_sol2)
            q1 = q1_sol1 if err1 <= err2 else q1_sol2
    Rz_inv_q1 = Rz(-q1)
    wrist_local = Rz_inv_q1 @ wrist_pos
    wx_local, wy_local, wz_local = wrist_local
    M = Rz_inv_q1 @ R_wrist
    q5 = math.atan2(M[1, 0], M[1, 1])
    M_no5 = M @ Rz(-q5)
    q6 = math.atan2(-M_no5[0, 2], M_no5[2, 2])
    M_no56 = M_no5 @ Ry(-q6)
    phi = math.atan2(M_no56[0, 2], M_no56[2, 2])
    target_234_x = wx_local - L3 * math.sin(phi) * math.cos(q6)
    target_234_z = wz_local - L3 * math.cos(phi) * math.cos(q6)
    r_squared = target_234_x ** 2 + target_234_z ** 2
    r = math.sqrt(r_squared)
    cos_q3 = (r_squared - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_pos = math.acos(cos_q3)
    q3_neg = -q3_pos

    def calculate_q2_q4(q3_val):
        theta = math.atan2(target_234_x, target_234_z)
        beta = math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q2_val = theta - beta
        q4_val = phi - (q2_val + q3_val)
        x_calc = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L3 * math.sin(phi) * math.cos(q6)
        z_calc = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L3 * math.cos(phi) * math.cos(q6)
        error = (x_calc - wx_local) ** 2 + (z_calc - wz_local) ** 2
        return (q2_val, q4_val, error)
    q2_pos, q4_pos, err_pos = calculate_q2_q4(q3_pos)
    q2_neg, q4_neg, err_neg = calculate_q2_q4(q3_neg)
    if err_pos <= err_neg:
        q2, q3, q4 = (q2_pos, q3_pos, q4_pos)
    else:
        q2, q3, q4 = (q2_neg, q3_neg, q4_neg)

    def check_alternate_solution():
        alt_q1 = normalize_angle(q1 + math.pi)
        alt_Rz_inv = Rz(-alt_q1)
        alt_wrist_local = alt_Rz_inv @ wrist_pos
        alt_wx, alt_wy, alt_wz = alt_wrist_local
        alt_M = alt_Rz_inv @ R_wrist
        alt_q5 = math.atan2(alt_M[1, 0], alt_M[1, 1])
        alt_M_no5 = alt_M @ Rz(-alt_q5)
        alt_q6 = math.atan2(-alt_M_no5[0, 2], alt_M_no5[2, 2])
        alt_M_no56 = alt_M_no5 @ Ry(-alt_q6)
        alt_phi = math.atan2(alt_M_no56[0, 2], alt_M_no56[2, 2])
        alt_target_x = alt_wx - L3 * math.sin(alt_phi) * math.cos(alt_q6)
        alt_target_z = alt_wz - L3 * math.cos(alt_phi) * math.cos(alt_q6)
        alt_r_squared = alt_target_x ** 2 + alt_target_z ** 2
        alt_cos_q3 = (alt_r_squared - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        alt_cos_q3 = max(-1.0, min(1.0, alt_cos_q3))
        if abs(alt_cos_q3 - 1.0) < 1e-06:
            alt_q3 = 0.0
        elif abs(alt_cos_q3 + 1.0) < 1e-06:
            alt_q3 = math.pi
        else:
            alt_q3_pos = math.acos(alt_cos_q3)
            alt_q3_neg = -alt_q3_pos
            alt_q2_pos, alt_q4_pos, err_pos = calculate_alt_q2_q4(alt_q3_pos, alt_target_x, alt_target_z, alt_phi, alt_q6)
            alt_q2_neg, alt_q4_neg, err_neg = calculate_alt_q2_q4(alt_q3_neg, alt_target_x, alt_target_z, alt_phi, alt_q6)
            if err_pos <= err_neg:
                alt_q2, alt_q3, alt_q4 = (alt_q2_pos, alt_q3_pos, alt_q4_pos)
            else:
                alt_q2, alt_q3, alt_q4 = (alt_q2_neg, alt_q3_neg, alt_q4_neg)
        return (alt_q1, alt_q2, alt_q3, alt_q4, alt_q5, alt_q6)

    def calculate_alt_q2_q4(q3_val, target_x, target_z, phi_val, q6_val):
        theta = math.atan2(target_x, target_z)
        beta = math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q2_val = theta - beta
        q4_val = phi_val - (q2_val + q3_val)
        x_calc = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L3 * math.sin(phi_val) * math.cos(q6_val)
        z_calc = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L3 * math.cos(phi_val) * math.cos(q6_val)
        error = (x_calc - target_x) ** 2 + (z_calc - target_z) ** 2
        return (q2_val, q4_val, error)
    if abs(q3) < 1e-06:
        q3 = 0
    q1 = normalize_angle(q1)
    q2 = normalize_angle(q2)
    q3 = normalize_angle(q3)
    q4 = normalize_angle(q4)
    q5 = normalize_angle(q5)
    q6 = normalize_angle(q6)

    def test_solution(q1, q2, q3, q4, q5, q6):
        T_01 = np.eye(4)
        T_01[:3, :3] = Rz(q1)
        T_12 = np.eye(4)
        T_12[1, 3] = Y2
        T_12[:3, :3] = Ry(q2)
        T_23 = np.eye(4)
        T_23[1, 3] = Y3
        T_23[2, 3] = L1
        T_23[:3, :3] = Ry(q3)
        T_34 = np.eye(4)
        T_34[2, 3] = L2
        T_34[:3, :3] = Ry(q4)
        T_45 = np.eye(4)
        T_45[1, 3] = Y5
        T_45[:3, :3] = Rz(q5)
        T_56 = np.eye(4)
        T_56[2, 3] = L3
        T_56[:3, :3] = Ry(q6)
        T_6T = np.eye(4)
        T_6T[1, 3] = TCP_Y
        T_6T[:3, :3] = Rz(math.pi / 2)
        T_0T = T_01 @ T_12 @ T_23 @ T_34 @ T_45 @ T_56 @ T_6T
        calc_pos = T_0T[:3, 3]
        calc_R = T_0T[:3, :3]
        pos_error = np.linalg.norm(calc_pos - np.array([px, py, pz]))
        R_error = np.linalg.norm(calc_R - R_target, 'fro')
        return pos_error + R_error
    if abs(q3) < 0.01:
        q3_alt = 0
        q2_alt = normalize_angle(theta)
        q4_alt = normalize_angle(phi - q2_alt)
        err_alt = test_solution(q1, q2_alt, q3_alt, q4_alt, q5, q6)
        err_orig = test_solution(q1, q2, q3, q4, q5, q6)
        if err_alt < err_orig:
            q2, q3, q4 = (q2_alt, q3_alt, q4_alt)
    q1 = normalize_angle(q1)
    q2 = normalize_angle(q2)
    q3 = normalize_angle(q3)
    q4 = normalize_angle(q4)
    q5 = normalize_angle(q5)
    q6 = normalize_angle(q6)
    return (q1, q2, q3, q4, q5, q6)