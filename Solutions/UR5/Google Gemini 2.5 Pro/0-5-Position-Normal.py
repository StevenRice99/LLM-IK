import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    px, py, pz = p
    d1 = 0.13585
    a2 = -0.1197
    d2_z = 0.425
    d3_z = 0.39225
    a4_y = 0.093
    d5_z = 0.09465
    d_tcp_y = 0.0823

    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))
    q4 = 0.0
    q6 = 0.0
    R_xy = math.sqrt(px ** 2 + py ** 2)
    K_const_Y = a4_y + a2 + d1
    c5 = 0.0
    min_Y_base_abs = abs(K_const_Y - d_tcp_y)
    if R_xy >= K_const_Y:
        c5 = 0.0
    elif R_xy >= min_Y_base_abs:
        if abs(d_tcp_y) < 1e-09:
            c5 = 0.0
        else:
            c5 = (R_xy - K_const_Y) / d_tcp_y
    else:
        c5 = -1.0
    c5 = clamp(c5, -1.0, 1.0)
    q5 = math.acos(c5)
    s5 = math.sin(q5)
    V_eff_x = -d_tcp_y * s5
    V_eff_y = d_tcp_y * c5 + a4_y
    V_eff_z = d3_z + d5_z
    Y_base_for_q1 = V_eff_y + a2 + d1
    asin_arg_q1 = 0.0
    if R_xy > 1e-09:
        asin_arg_q1 = Y_base_for_q1 / R_xy
    elif abs(Y_base_for_q1) < 1e-09:
        asin_arg_q1 = 0.0
    q1 = math.atan2(py, px) - math.asin(clamp(asin_arg_q1, -1.0, 1.0))
    c_q1 = math.cos(q1)
    s_q1 = math.sin(q1)
    X_J1_frame = px * c_q1 + py * s_q1
    Z_J1_frame = pz
    LHS_q3_sq = X_J1_frame ** 2 + Z_J1_frame ** 2
    K0_q3 = V_eff_x ** 2 + V_eff_z ** 2 + d2_z ** 2
    Kc_q3 = 2 * d2_z * V_eff_z
    Ks_q3 = -2 * d2_z * V_eff_x
    D_q3 = LHS_q3_sq - K0_q3
    den_sqrt_q3_sq = Ks_q3 ** 2 + Kc_q3 ** 2
    if den_sqrt_q3_sq < 1e-12:
        if abs(D_q3) < 1e-09:
            q3 = 0.0
        else:
            q3 = 0.0
    else:
        den_sqrt_q3 = math.sqrt(den_sqrt_q3_sq)
        acos_arg_q3 = clamp(D_q3 / den_sqrt_q3, -1.0, 1.0)
        phi_q3 = math.acos(acos_arg_q3)
        atan_term_q3 = math.atan2(Ks_q3, Kc_q3)
        q3 = atan_term_q3 - phi_q3
    s_q3 = math.sin(q3)
    c_q3 = math.cos(q3)
    A_for_q2 = V_eff_x * c_q3 + V_eff_z * s_q3
    B_for_q2 = -V_eff_x * s_q3 + V_eff_z * c_q3 + d2_z
    if LHS_q3_sq < 1e-09:
        q2 = 0.0
    else:
        q2_num_atan = X_J1_frame * B_for_q2 - Z_J1_frame * A_for_q2
        q2_den_atan = X_J1_frame * A_for_q2 + Z_J1_frame * B_for_q2
        q2 = math.atan2(q2_num_atan, q2_den_atan)
    return (q1, q2, q3, q4, q5, q6)