import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    px, py, pz = p
    q6 = 0.0
    L1_E10 = 0.425
    L2_E10 = 0.39225
    L3_E10 = 0.09465
    y_offset_E10 = -0.1197 + 0.093
    tcp_y_offset_E10 = 0.0823
    d_J1_J2_y = 0.13585
    best_overall_solution = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_overall_error = float('inf')
    num_q1_steps = 72
    for i in range(num_q1_steps):
        q1_candidate = i * 2 * math.pi / num_q1_steps - math.pi
        c1 = math.cos(q1_candidate)
        s1 = math.sin(q1_candidate)
        x_j1_rot = c1 * px + s1 * py
        y_j1_rot = -s1 * px + c1 * py
        z_j1_rot = pz
        x_j2_target = x_j1_rot
        y_j2_target = y_j1_rot - d_J1_J2_y
        z_j2_target = z_j1_rot
        current_E10_best_error = float('inf')
        current_E10_best_solution_joints = None
        cos_q4_E10_val = (y_j2_target - y_offset_E10) / tcp_y_offset_E10
        if abs(cos_q4_E10_val) > 1.0:
            if abs(cos_q4_E10_val) - 1.0 < 1e-09:
                cos_q4_E10_val = math.copysign(1.0, cos_q4_E10_val)
            else:
                continue
        q4_E10_angle_abs = math.acos(cos_q4_E10_val)
        q4_E10_candidates = [q4_E10_angle_abs, -q4_E10_angle_abs]
        psi_E10 = math.atan2(x_j2_target, z_j2_target)
        for q4_E10_sol in q4_E10_candidates:
            q5_current_sol = q4_E10_sol
            d_val_E10 = tcp_y_offset_E10 * math.sin(q5_current_sol)
            L_eff_E10 = math.sqrt(L3_E10 ** 2 + d_val_E10 ** 2)
            phi_val_E10 = math.atan2(d_val_E10, L3_E10)
            for T_offset_choice_E10 in [0, math.pi]:
                T_candidate_E10 = psi_E10 + T_offset_choice_E10
                S_E10 = T_candidate_E10 + phi_val_E10
                W_x_E10 = x_j2_target - L_eff_E10 * math.sin(T_candidate_E10)
                W_z_E10 = z_j2_target - L_eff_E10 * math.cos(T_candidate_E10)
                r_w_sq_E10 = W_x_E10 ** 2 + W_z_E10 ** 2
                if r_w_sq_E10 > (L1_E10 + L2_E10) ** 2 + 1e-09 or r_w_sq_E10 < (L1_E10 - L2_E10) ** 2 - 1e-09:
                    continue
                cos_q2_E10_val_num = r_w_sq_E10 - L1_E10 ** 2 - L2_E10 ** 2
                cos_q2_E10_val_den = 2 * L1_E10 * L2_E10
                if abs(cos_q2_E10_val_den) < 1e-12:
                    continue
                cos_q2_E10_val = cos_q2_E10_val_num / cos_q2_E10_val_den
                cos_q2_E10_val = max(-1.0, min(1.0, cos_q2_E10_val))
                q2_E10_angle_abs = math.acos(cos_q2_E10_val)
                for q2_E10_sign in [1, -1]:
                    q2_E10_sol = q2_E10_sign * q2_E10_angle_abs
                    den_delta_E10 = L1_E10 + L2_E10 * math.cos(q2_E10_sol)
                    num_delta_E10 = L2_E10 * math.sin(q2_E10_sol)
                    if abs(den_delta_E10) < 1e-09 and abs(num_delta_E10) < 1e-09:
                        delta_E10 = 0
                    elif abs(den_delta_E10) < 1e-09:
                        delta_E10 = math.copysign(math.pi / 2, num_delta_E10) if num_delta_E10 != 0 else 0
                    else:
                        delta_E10 = math.atan2(num_delta_E10, den_delta_E10)
                    theta_w_E10 = math.atan2(W_x_E10, W_z_E10)
                    q1_E10_sol = theta_w_E10 - delta_E10
                    q3_E10_sol = S_E10 - (q1_E10_sol + q2_E10_sol)
                    S_check_E10 = q1_E10_sol + q2_E10_sol + q3_E10_sol
                    d_check_E10 = tcp_y_offset_E10 * math.sin(q5_current_sol)
                    x_fk_E10 = L1_E10 * math.sin(q1_E10_sol) + L2_E10 * math.sin(q1_E10_sol + q2_E10_sol) + L3_E10 * math.sin(S_check_E10) - d_check_E10 * math.cos(S_check_E10)
                    y_fk_E10 = y_offset_E10 + tcp_y_offset_E10 * math.cos(q5_current_sol)
                    z_fk_E10 = L1_E10 * math.cos(q1_E10_sol) + L2_E10 * math.cos(q1_E10_sol + q2_E10_sol) + L3_E10 * math.cos(S_check_E10) + d_check_E10 * math.sin(S_check_E10)
                    error_E10_sq = (x_fk_E10 - x_j2_target) ** 2 + (y_fk_E10 - y_j2_target) ** 2 + (z_fk_E10 - z_j2_target) ** 2
                    if error_E10_sq < current_E10_best_error:
                        current_E10_best_error = error_E10_sq
                        current_E10_best_solution_joints = (q1_E10_sol, q2_E10_sol, q3_E10_sol, q5_current_sol)
        if current_E10_best_solution_joints is not None:
            q2_sol, q3_sol, q4_sol, q5_sol = current_E10_best_solution_joints
            current_total_joints = (q1_candidate, q2_sol, q3_sol, q4_sol, q5_sol, q6)
            if current_E10_best_error < min_overall_error:
                min_overall_error = current_E10_best_error
                best_overall_solution = current_total_joints

    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    return tuple((normalize_angle(j) for j in best_overall_solution))