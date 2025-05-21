import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    px, py, pz = p
    rx_in, ry_in, rz_in = r
    l2z = 0.425
    l3z = 0.39225
    TCP_offset_y = 0.093
    crx = math.cos(rx_in)
    srx = math.sin(rx_in)
    cry = math.cos(ry_in)
    sry = math.sin(ry_in)
    crz = math.cos(rz_in)
    srz = math.sin(rz_in)
    r11 = crz * cry
    r12 = crz * sry * srx - srz * crx
    r22 = srz * sry * srx + crz * crx
    r31 = -sry
    r32 = cry * srx
    r33 = cry * crx
    pwx = px - r12 * TCP_offset_y
    pwy = py - r22 * TCP_offset_y
    pwz = pz - r32 * TCP_offset_y
    q1 = math.atan2(-r12, r22)
    q_sum_angles = math.atan2(-r31, r33)
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    pwx_F1 = c1 * pwx + s1 * pwy
    pwz_F1 = pwz
    vx_planar = pwx_F1
    vz_planar = pwz_F1
    cos_q3_num = vx_planar ** 2 + vz_planar ** 2 - l2z ** 2 - l3z ** 2
    cos_q3_den = 2 * l2z * l3z
    if abs(cos_q3_den) < 1e-12:
        cos_q3_clamped = 2.0
    else:
        cos_q3_clamped = cos_q3_num / cos_q3_den
    if cos_q3_clamped > 1.0:
        cos_q3_clamped = 1.0
    elif cos_q3_clamped < -1.0:
        cos_q3_clamped = -1.0
    q3_A = math.acos(cos_q3_clamped)
    q3_B = -math.acos(cos_q3_clamped)
    solutions_data = []
    for q3_choice in [q3_A, q3_B]:
        s3 = math.sin(q3_choice)
        c3 = cos_q3_clamped
        X_arm = l3z * s3
        Z_arm = l2z + l3z * c3
        den_q2_calc = vx_planar ** 2 + vz_planar ** 2
        if abs(den_q2_calc) < 1e-12:
            s2_val = 0.0
            c2_val = 1.0
        else:
            s2_val = (vx_planar * Z_arm - vz_planar * X_arm) / den_q2_calc
            c2_val = (vx_planar * X_arm + vz_planar * Z_arm) / den_q2_calc
        q2_choice = math.atan2(s2_val, c2_val)
        q4_choice = q_sum_angles - q2_choice - q3_choice
        solutions_data.append({'q1': q1, 'q2': q2_choice, 'q3': q3_choice, 'q4': q4_choice, 'is_A': q3_choice == q3_A})
    best_solution_idx = -1
    min_abs_norm_q4 = float('inf')
    for idx, sol_data in enumerate(solutions_data):
        q4_norm = math.atan2(math.sin(sol_data['q4']), math.cos(sol_data['q4']))
        current_abs_norm_q4 = abs(q4_norm)
        if current_abs_norm_q4 < min_abs_norm_q4:
            min_abs_norm_q4 = current_abs_norm_q4
            best_solution_idx = idx
        elif abs(current_abs_norm_q4 - min_abs_norm_q4) < 1e-09:
            if solutions_data[idx]['is_A']:
                best_solution_idx = idx
    final_sol = solutions_data[best_solution_idx]
    return (final_sol['q1'], final_sol['q2'], final_sol['q3'], final_sol['q4'])