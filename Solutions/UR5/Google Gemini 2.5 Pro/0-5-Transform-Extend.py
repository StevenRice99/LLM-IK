import math
import numpy as np

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry_mat = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz_mat = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz_mat @ Ry_mat @ Rx

def _matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular_threshold = 1e-06
    if sy > singular_threshold:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch_plus_half_pi = R[2, 0] < -0.99999
        pitch_minus_half_pi = R[2, 0] > 0.99999
        if pitch_plus_half_pi:
            pitch = math.pi / 2.0
            yaw = 0.0
            roll = math.atan2(R[0, 1], R[1, 1])
        elif pitch_minus_half_pi:
            pitch = -math.pi / 2.0
            yaw = 0.0
            roll = math.atan2(-R[0, 1], R[1, 1])
        else:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy if sy > 1e-09 else 1e-09)
            yaw = math.atan2(R[1, 0], R[0, 0])
    return (roll, pitch, yaw)

def _Ry_matrix(angle: float) -> np.ndarray:
    c, s = (math.cos(angle), math.sin(angle))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def _Rz_matrix(angle: float) -> np.ndarray:
    c, s = (math.cos(angle), math.sin(angle))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def _ik_5dof(p_5dof_tuple: tuple[float, float, float], r_5dof_tuple: tuple[float, float, float]) -> tuple[tuple[float, float, float, float, float], float]:
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    pi = math.pi
    tol = 1e-07
    p_x, p_y, p_z = p_5dof_tuple
    roll, pitch, yaw = r_5dof_tuple
    R_des = _rpy_to_matrix(roll, pitch, yaw)
    r_xy = math.sqrt(p_x ** 2 + p_y ** 2)
    q1_candidates_5dof = []
    if r_xy < tol:
        if abs(y_const) < tol:
            theta = 0.0
            a = 0.0
        else:
            raise ValueError('5-DOF: Unreachable (P_wc on Z-axis, y_const != 0)')
    else:
        theta = math.atan2(p_y, p_x)
        ratio = y_const / r_xy
        if abs(ratio) > 1.0 + tol:
            raise ValueError(f'5-DOF: Unreachable q1 (asin out of bounds: {ratio})')
        ratio = max(-1.0, min(1.0, ratio))
        a = math.asin(ratio)
    q1_candidates_5dof.extend([theta - a, theta - (pi - a)])
    valid_q1_solutions = []
    for q1_val_cand in q1_candidates_5dof:
        cos_q1_cand, sin_q1_cand = (math.cos(q1_val_cand), math.sin(q1_val_cand))
        Rz_neg_q1_cand = np.array([[cos_q1_cand, sin_q1_cand, 0], [-sin_q1_cand, cos_q1_cand, 0], [0, 0, 1]], dtype=float)
        M_val = Rz_neg_q1_cand @ R_des
        valid_q1_solutions.append({'q1': q1_val_cand, 'M': M_val, 'err_m12': abs(M_val[1, 2])})
    if not valid_q1_solutions:
        raise ValueError('5-DOF: No valid q1 candidates found.')
    valid_q1_solutions.sort(key=lambda s: s['err_m12'])
    chosen_q1_solution = valid_q1_solutions[0]
    q1 = chosen_q1_solution['q1']
    M = chosen_q1_solution['M']
    cos_q1, sin_q1 = (math.cos(q1), math.sin(q1))
    Rz_neg_q1_mat = np.array([[cos_q1, sin_q1, 0], [-sin_q1, cos_q1, 0], [0, 0, 1]], dtype=float)
    p_vec = np.array([p_x, p_y, p_z], dtype=float)
    p_bar = Rz_neg_q1_mat @ p_vec
    p_bar_x, _, p_bar_z = p_bar
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = math.atan2(M[1, 0], M[1, 1])
    P_x_planar = p_bar_x - L_tcp * math.sin(phi)
    P_z_planar = p_bar_z - L_tcp * math.cos(phi)
    r2_planar_sq = P_x_planar ** 2 + P_z_planar ** 2
    L_sum_sq = (L1 + L2) ** 2
    L_diff_sq = (L1 - L2) ** 2
    cos_q3_val_num = r2_planar_sq - L1 ** 2 - L2 ** 2
    cos_q3_val_den = 2 * L1 * L2
    if abs(cos_q3_val_den) < tol:
        raise ValueError('5-DOF: L1 or L2 is zero in 2R arm.')
    cos_q3_val = max(-1.0, min(1.0, cos_q3_val_num / cos_q3_val_den))
    q3_candidateA = math.acos(cos_q3_val)
    q3_candidateB = -q3_candidateA
    solutions_2R = []
    for q3_val in [q3_candidateA, q3_candidateB]:
        den_q2_beta = L1 + L2 * math.cos(q3_val)
        beta = 0.0
        if not (abs(den_q2_beta) < tol and abs(L2 * math.sin(q3_val)) < tol):
            beta = math.atan2(L2 * math.sin(q3_val), den_q2_beta)
        q2_val = math.atan2(P_x_planar, P_z_planar) - beta
        q4_val = phi - (q2_val + q3_val)
        calc_x_rec = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L_tcp * math.sin(phi)
        calc_z_rec = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L_tcp * math.cos(phi)
        err_val = math.sqrt((calc_x_rec - p_bar_x) ** 2 + (calc_z_rec - p_bar_z) ** 2)
        solutions_2R.append({'q2': q2_val, 'q3': q3_val, 'q4': q4_val, 'err': err_val})
    if not solutions_2R:
        raise ValueError('5-DOF: No solution found in planar subproblem.')
    solutions_2R.sort(key=lambda x: x['err'])
    best_2R = solutions_2R[0]
    return ((q1, best_2R['q2'], best_2R['q3'], best_2R['q4'], q5), best_2R['err'])

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    d_tcp_offset_vec = np.array([0, 0.0823, 0], dtype=float)
    y_const_for_q1 = 0.13585 - 0.1197 + 0.093
    pi = math.pi
    tol = 1e-07
    R_tcp_fixed_inv = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    p_target_vec = np.array(p, dtype=float)
    R_target = _rpy_to_matrix(r[0], r[1], r[2])
    R_0_6 = R_target @ R_tcp_fixed_inv
    P_wc_vec = p_target_vec - R_0_6 @ d_tcp_offset_vec
    P_wc_x, P_wc_y = (P_wc_vec[0], P_wc_vec[1])
    r_xy_q1 = math.sqrt(P_wc_x ** 2 + P_wc_y ** 2)
    q1_cand_forms_6dof = []
    if r_xy_q1 < tol:
        if abs(y_const_for_q1) < tol:
            theta_q1 = 0.0
            a_q1 = 0.0
        else:
            raise ValueError('IK: Unreachable q1 (P_wc on Z, y_const != 0)')
    else:
        theta_q1 = math.atan2(P_wc_y, P_wc_x)
        ratio_q1 = y_const_for_q1 / r_xy_q1
        if abs(ratio_q1) > 1.0 + tol:
            raise ValueError(f'IK: Unreachable q1 (asin out of bounds: {ratio_q1})')
        ratio_q1 = max(-1.0, min(1.0, ratio_q1))
        a_q1 = math.asin(ratio_q1)
    q1_cand_forms_6dof.extend([theta_q1 - a_q1, theta_q1 - (pi - a_q1)])
    all_consistent_solutions = []
    for q1_cand in q1_cand_forms_6dof:
        R_z_neg_q1 = _Rz_matrix(-q1_cand)
        M_known_part = R_z_neg_q1 @ R_0_6
        mkp_10, mkp_12 = (M_known_part[1, 0], M_known_part[1, 2])
        current_q6_candidates = []
        if abs(mkp_10) < tol and abs(mkp_12) < tol:
            current_q6_candidates.append(0.0)
        else:
            q6_base = math.atan2(mkp_12, mkp_10)
            current_q6_candidates.append(q6_base)
            current_q6_candidates.append(math.fmod(q6_base + pi + pi, 2 * pi) - pi)
        for q6_c in current_q6_candidates:
            R_0_5_frame = R_0_6 @ _Ry_matrix(-q6_c)
            try:
                r_5dof_tuple = _matrix_to_rpy(R_0_5_frame)
            except ValueError:
                continue
            P_wc_tuple = tuple(P_wc_vec)
            try:
                (q1_s, q2_s, q3_s, q4_s, q5_s), err_5dof = _ik_5dof(P_wc_tuple, r_5dof_tuple)
            except ValueError:
                continue
            q1_s_norm = math.fmod(q1_s + pi, 2 * pi) - pi
            q1_cand_norm = math.fmod(q1_cand + pi, 2 * pi) - pi
            diff_q1 = abs(q1_s_norm - q1_cand_norm)
            if diff_q1 > pi - tol:
                diff_q1 = 2 * pi - diff_q1
            if diff_q1 < tol:
                result_angles = [q1_s, q2_s, q3_s, q4_s, q5_s, q6_c]
                normalized_result = []
                for angle_val in result_angles:
                    norm_angle = math.fmod(angle_val + pi, 2 * pi) - pi
                    if abs(angle_val - pi) < tol:
                        normalized_result.append(pi)
                    elif abs(angle_val + pi) < tol:
                        normalized_result.append(-pi)
                    else:
                        normalized_result.append(norm_angle)
                all_consistent_solutions.append({'angles': tuple(normalized_result), 'error': err_5dof, 'q3_val': q3_s})
    if not all_consistent_solutions:
        raise ValueError('6-DOF IK: No consistent solution found across all branches.')
    all_consistent_solutions.sort(key=lambda x: (x['error'], abs(x['q3_val']) < 0.0001))
    return all_consistent_solutions[0]['angles']