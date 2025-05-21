import math

def _normalize_angle(angle):
    """Normalize angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _solve_j2_j3_v3(x_pt, z_pt, t_sum_principal, L2z, L3z, abs_tol=1e-05):
    """
    Solves the J2-J3 planar subproblem, considering 2pi variations for t_sum.
    Returns (theta2, theta3, is_primary_solution_flag) or (None, None, False)
    """
    d_sq_planar = x_pt ** 2 + z_pt ** 2
    val_for_acos_theta3_num = d_sq_planar - L2z ** 2 - L3z ** 2
    val_for_acos_theta3_den = 2 * L2z * L3z
    if abs(val_for_acos_theta3_den) < 1e-09:
        return (None, None, False)
    val_for_acos_theta3 = val_for_acos_theta3_num / val_for_acos_theta3_den
    clamped_val = max(min(val_for_acos_theta3, 1.0), -1.0)
    theta3_from_acos_raw_val = math.acos(clamped_val)
    base_theta3_candidates = [-theta3_from_acos_raw_val, theta3_from_acos_raw_val]
    k_sum_options = [0, -1, 1]
    for t3_cand_raw in base_theta3_candidates:
        for k_sum in k_sum_options:
            current_t_sum = t_sum_principal + k_sum * 2 * math.pi
            t2_cand = current_t_sum - t3_cand_raw
            x_calc = L2z * math.sin(t2_cand) + L3z * math.sin(current_t_sum)
            z_calc = L2z * math.cos(t2_cand) + L3z * math.cos(current_t_sum)
            if math.isclose(x_calc, x_pt, abs_tol=abs_tol) and math.isclose(z_calc, z_pt, abs_tol=abs_tol):
                return (t2_cand, t3_cand_raw, True)
    for k_sum in k_sum_options:
        current_t_sum = t_sum_principal + k_sum * 2 * math.pi
        term_x = x_pt - L3z * math.sin(current_t_sum)
        term_z = z_pt - L3z * math.cos(current_t_sum)
        t2_fallback = math.atan2(term_x, term_z)
        t3_fallback = current_t_sum - t2_fallback
        x_calc_fb = L2z * math.sin(t2_fallback) + L3z * math.sin(current_t_sum)
        z_calc_fb = L2z * math.cos(t2_fallback) + L3z * math.cos(current_t_sum)
        if math.isclose(x_calc_fb, x_pt, abs_tol=abs_tol) and math.isclose(z_calc_fb, z_pt, abs_tol=abs_tol):
            return (t2_fallback, t3_fallback, False)
    return (None, None, False)

def inverse_kinematics(p_global: tuple[float, float, float], r_global_rpy_zyx: tuple[float, float, float]) -> tuple[float, float, float]:
    px, py, pz = p_global
    rx_g, ry_g, rz_g = r_global_rpy_zyx
    d1y = 0.13585
    d2y = -0.1197
    C_offset = d1y + d2y
    L2z = 0.425
    L3z = 0.39225
    R_val_sq = px ** 2 + py ** 2
    if R_val_sq < 1e-12:
        R_val = 1e-06
    else:
        R_val = math.sqrt(R_val_sq)
    alpha_angle = math.atan2(-px, py)
    cos_phi_val_arg = C_offset / R_val
    clamped_cos_phi_val = max(min(cos_phi_val_arg, 1.0), -1.0)
    phi_angle = math.acos(clamped_cos_phi_val)
    theta1_options = [alpha_angle - phi_angle, alpha_angle + phi_angle]
    all_found_solutions = []
    for theta1 in theta1_options:
        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        x_planar_target = px * c1 + py * s1
        z_planar_target = pz
        crx, srx = (math.cos(rx_g), math.sin(rx_g))
        cry, sry = (math.cos(ry_g), math.sin(ry_g))
        crz, srz = (math.cos(rz_g), math.sin(rz_g))
        R00_g = crz * cry
        R02_g = crz * sry * crx + srz * srx
        R10_g = srz * cry
        R12_g = srz * sry * crx - crz * srx
        R_target_F1_00 = c1 * R00_g + s1 * R10_g
        R_target_F1_02 = c1 * R02_g + s1 * R12_g
        theta_sum_j2j3_principal = math.atan2(R_target_F1_02, R_target_F1_00)
        theta2, theta3, is_primary = _solve_j2_j3_v3(x_planar_target, z_planar_target, theta_sum_j2j3_principal, L2z, L3z)
        if theta2 is not None and theta3 is not None:
            all_found_solutions.append({'t1': theta1, 't2': theta2, 't3': theta3, 'is_primary': is_primary})
    if not all_found_solutions:
        return (0.0, 0.0, 0.0)
    primary_sols = [s for s in all_found_solutions if s['is_primary']]
    if primary_sols:
        chosen_sol = primary_sols[0]
        return (chosen_sol['t1'], chosen_sol['t2'], chosen_sol['t3'])
    else:
        chosen_sol = all_found_solutions[0]
        return (chosen_sol['t1'], chosen_sol['t2'], chosen_sol['t3'])