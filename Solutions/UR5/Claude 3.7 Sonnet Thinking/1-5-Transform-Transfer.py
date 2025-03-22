def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
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
    y_offset = -0.1197 + 0.093
    tcp_y_offset = 0.0823
    x_target, y_target, z_target = p
    rx_target, ry_target, rz_target = r

    def normalize(angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def rpy_to_rot_matrix(rx, ry, rz):
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def joint_rot_matrices(q1, q2, q3, q4, q5):
        R1 = np.array([[math.cos(q1), 0, math.sin(q1)], [0, 1, 0], [-math.sin(q1), 0, math.cos(q1)]])
        R2 = np.array([[math.cos(q2), 0, math.sin(q2)], [0, 1, 0], [-math.sin(q2), 0, math.cos(q2)]])
        R3 = np.array([[math.cos(q3), 0, math.sin(q3)], [0, 1, 0], [-math.sin(q3), 0, math.cos(q3)]])
        R4 = np.array([[math.cos(q4), -math.sin(q4), 0], [math.sin(q4), math.cos(q4), 0], [0, 0, 1]])
        R5 = np.array([[math.cos(q5), 0, math.sin(q5)], [0, 1, 0], [-math.sin(q5), 0, math.cos(q5)]])
        Rtcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        return (R1, R2, R3, R4, R5, Rtcp)

    def forward_kinematics(q1, q2, q3, q4, q5):
        S = q1 + q2 + q3
        d = tcp_y_offset * math.sin(q4)
        x = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(S) - d * math.cos(S)
        z = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(S) + d * math.sin(S)
        y = y_offset + tcp_y_offset * math.cos(q4)
        R1, R2, R3, R4, R5, Rtcp = joint_rot_matrices(q1, q2, q3, q4, q5)
        R = R1 @ R2 @ R3 @ R4 @ R5 @ Rtcp
        if abs(R[2, 0]) >= 0.99999:
            ry = -math.pi / 2 if R[2, 0] > 0 else math.pi / 2
            rz = 0
            rx = math.atan2(-R[0, 1], R[1, 1])
        else:
            ry = math.asin(-R[2, 0])
            rx = math.atan2(R[2, 1], R[2, 2])
            rz = math.atan2(R[1, 0], R[0, 0])
        return ((x, y, z), (rx, ry, rz), R)
    R_target = rpy_to_rot_matrix(rx_target, ry_target, rz_target)
    C = (y_target - y_offset) / tcp_y_offset
    C = max(min(C, 1.0), -1.0)
    q4_candidates = [math.acos(C), -math.acos(C)]
    psi = math.atan2(x_target, z_target)
    best_error = float('inf')
    best_solution = None
    for q4 in q4_candidates:
        d = tcp_y_offset * math.sin(q4)
        L_eff = math.sqrt(L3 ** 2 + d ** 2)
        phi = math.atan2(d, L3)
        for T in [psi, psi + math.pi]:
            S = T + phi
            W_x = x_target - L_eff * math.sin(T)
            W_z = z_target - L_eff * math.cos(T)
            r_w = math.hypot(W_x, W_z)
            if r_w > L1 + L2 or r_w < abs(L1 - L2):
                continue
            cos_q2 = (r_w ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            cos_q2 = max(min(cos_q2, 1.0), -1.0)
            for sign_q2 in [1, -1]:
                q2 = sign_q2 * math.acos(cos_q2)
                delta = math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
                theta_w = math.atan2(W_x, W_z)
                q1 = theta_w - delta
                q3 = S - (q1 + q2)
                for q5_base in [0, math.pi, -math.pi]:
                    for q5_offset in np.linspace(-math.pi, math.pi, 36):
                        q5 = q5_base + q5_offset
                        (x_fk, y_fk, z_fk), (rx_fk, ry_fk, rz_fk), R_fk = forward_kinematics(q1, q2, q3, q4, q5)
                        pos_error = math.sqrt((x_fk - x_target) ** 2 + (y_fk - y_target) ** 2 + (z_fk - z_target) ** 2)
                        ori_error = np.linalg.norm(R_fk - R_target, 'fro')
                        total_error = pos_error + 3.0 * ori_error
                        if total_error < best_error:
                            best_error = total_error
                            best_solution = (q1, q2, q3, q4, q5)
    if best_solution is None:
        raise ValueError('No valid IK solution found for the input target position and orientation.')
    return best_solution