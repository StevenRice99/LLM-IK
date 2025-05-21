import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """

    def normalize_angle(angle: float) -> float:
        res_angle = angle
        while res_angle > math.pi:
            res_angle -= 2.0 * math.pi
        while res_angle < -math.pi:
            res_angle += 2.0 * math.pi
        return res_angle

    def _solve_j2_to_j6_subchain(p_sub: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
        """
        Analytical closed–form inverse kinematics for a 5-DOF serial manipulator.
        This is taken from EXISTING 2, solving for our J2, J3, J4, J5, J6.
        The input p_sub is the target TCP position in J2's frame.
        Returns (q2, q3, q4, q5, q6) where q6 will be 0.0.
        """
        L1_sub = 0.425
        L2_sub = 0.39225
        L3_sub = 0.09465
        y_offset_links_sub = -0.1197 + 0.093
        tcp_y_contribution_factor_sub = 0.0823
        x_target_sub, y_target_sub, z_target_sub = p_sub

        def normalize_sub(angle: float) -> float:
            res_angle = angle
            while res_angle > math.pi:
                res_angle -= 2.0 * math.pi
            while res_angle < -math.pi:
                res_angle += 2.0 * math.pi
            return res_angle

        def fk_subchain(q1_s, q2_s, q3_s, q4_s):
            S_sum_y_rots = q1_s + q2_s + q3_s
            d_factor_val = tcp_y_contribution_factor_sub * math.sin(q4_s)
            x_fk = L1_sub * math.sin(q1_s) + L2_sub * math.sin(q1_s + q2_s) + L3_sub * math.sin(S_sum_y_rots) - d_factor_val * math.cos(S_sum_y_rots)
            z_fk = L1_sub * math.cos(q1_s) + L2_sub * math.cos(q1_s + q2_s) + L3_sub * math.cos(S_sum_y_rots) + d_factor_val * math.sin(S_sum_y_rots)
            y_fk = y_offset_links_sub + tcp_y_contribution_factor_sub * math.cos(q4_s)
            return (x_fk, y_fk, z_fk)
        cos_q4_s_val = (y_target_sub - y_offset_links_sub) / tcp_y_contribution_factor_sub
        cos_q4_s_val = max(min(cos_q4_s_val, 1.0), -1.0)
        q4_s_candidates = [math.acos(cos_q4_s_val), -math.acos(cos_q4_s_val)]
        psi_xz_plane = math.atan2(x_target_sub, z_target_sub)
        best_error_val = float('inf')
        best_solution_subchain = None
        for q4_s_candidate_val in q4_s_candidates:
            d_val_calc = tcp_y_contribution_factor_sub * math.sin(q4_s_candidate_val)
            L_eff_calc = math.sqrt(L3_sub ** 2 + d_val_calc ** 2)
            phi_eff_angle = math.atan2(d_val_calc, L3_sub)
            for T_candidate_base_angle in [psi_xz_plane, psi_xz_plane + math.pi]:
                T_candidate_val = normalize_sub(T_candidate_base_angle)
                S_total_y_rots_candidate = T_candidate_val + phi_eff_angle
                W_x_val = x_target_sub - L_eff_calc * math.sin(T_candidate_val)
                W_z_val = z_target_sub - L_eff_calc * math.cos(T_candidate_val)
                r_w_val = math.hypot(W_x_val, W_z_val)
                epsilon_reach = 1e-09
                if r_w_val > L1_sub + L2_sub + epsilon_reach or r_w_val < abs(L1_sub - L2_sub) - epsilon_reach:
                    continue
                cos_q2_s_val_num = r_w_val ** 2 - L1_sub ** 2 - L2_sub ** 2
                cos_q2_s_val_den = 2 * L1_sub * L2_sub
                if abs(cos_q2_s_val_den) < 1e-12:
                    continue
                cos_q2_s_val = cos_q2_s_val_num / cos_q2_s_val_den
                cos_q2_s_val = max(min(cos_q2_s_val, 1.0), -1.0)
                for elbow_sign_val in [1, -1]:
                    q2_s_candidate_val = elbow_sign_val * math.acos(cos_q2_s_val)
                    delta_angle = math.atan2(L2_sub * math.sin(q2_s_candidate_val), L1_sub + L2_sub * math.cos(q2_s_candidate_val))
                    theta_w_angle = math.atan2(W_x_val, W_z_val)
                    q1_s_candidate_val = theta_w_angle - delta_angle
                    q3_s_candidate_val = S_total_y_rots_candidate - (q1_s_candidate_val + q2_s_candidate_val)
                    x_fk_val, y_fk_val, z_fk_val = fk_subchain(q1_s_candidate_val, q2_s_candidate_val, q3_s_candidate_val, q4_s_candidate_val)
                    current_error = math.sqrt((x_fk_val - x_target_sub) ** 2 + (y_fk_val - y_target_sub) ** 2 + (z_fk_val - z_target_sub) ** 2)
                    if current_error < best_error_val:
                        best_error_val = current_error
                        best_solution_subchain = (q1_s_candidate_val, q2_s_candidate_val, q3_s_candidate_val, q4_s_candidate_val, 0.0)
        if best_solution_subchain is None:
            raise ValueError('Subchain solver: No valid IK solution found.')
        q2_f, q3_f, q4_f, q5_f, q6_f = best_solution_subchain
        return (normalize_sub(q2_f), normalize_sub(q3_f), normalize_sub(q4_f), normalize_sub(q5_f), normalize_sub(q6_f))
    px, py, pz = p
    L_J1_J2y_offset = 0.13585
    q1 = math.atan2(-px, py)
    s1 = math.sin(q1)
    c1 = math.cos(q1)
    x_target_for_j2_subchain = px * c1 + py * s1
    y_target_for_j2_subchain = -px * s1 + py * c1 - L_J1_J2y_offset
    z_target_for_j2_subchain = pz
    p_subchain_target = (x_target_for_j2_subchain, y_target_for_j2_subchain, z_target_for_j2_subchain)
    q2, q3, q4, q5, q6 = _solve_j2_to_j6_subchain(p_subchain_target)
    return (normalize_angle(q1), q2, q3, q4, q5, q6)