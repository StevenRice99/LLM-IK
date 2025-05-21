def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    import numpy as np
    px, py, pz = p
    rx, ry, rz = r
    d2z = 0.425
    d3z = 0.39225
    d4y_tcp = 0.093
    twopi = 2 * np.pi
    joint_limit = twopi
    epsilon = 1e-09
    cr, sr = (np.cos(rx), np.sin(rx))
    cp, sp = (np.cos(ry), np.sin(ry))
    cy, sy = (np.cos(rz), np.sin(rz))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    xw = px - R_target[0, 1] * d4y_tcp
    yw = py - R_target[1, 1] * d4y_tcp
    zw = pz - R_target[2, 1] * d4y_tcp
    q1 = np.arctan2(-R_target[0, 1], R_target[1, 1])
    q_sum = np.arctan2(-R_target[2, 0], R_target[2, 2])
    c1, s1 = (np.cos(q1), np.sin(q1))
    P_1x = c1 * xw + s1 * yw
    P_1z = zw
    val_num = P_1z ** 2 + P_1x ** 2 - d2z ** 2 - d3z ** 2
    val_den = 2 * d2z * d3z
    cos_q3_val = 2.0
    if abs(val_den) > epsilon:
        cos_q3_val = val_num / val_den
    cos_q3_val = np.clip(cos_q3_val, -1.0, 1.0)
    q3_base_A = np.arccos(cos_q3_val)
    q3_base_B = -np.arccos(cos_q3_val)
    q3_options_ordered = [q3_base_A, q3_base_B, q3_base_A - twopi, q3_base_B + twopi]
    for q3_candidate in q3_options_ordered:
        sin_q3 = np.sin(q3_candidate)
        cos_q3 = np.cos(q3_candidate)
        q2_candidate = np.arctan2(P_1x, P_1z) - np.arctan2(d3z * sin_q3, d2z + d3z * cos_q3)
        q4_raw = q_sum - q2_candidate - q3_candidate
        q4_final = q4_raw
        while q4_final > joint_limit + epsilon:
            q4_final -= twopi
        while q4_final < -joint_limit - epsilon:
            q4_final += twopi
        if q4_final >= -joint_limit - epsilon and q4_final <= joint_limit + epsilon and (q3_candidate >= -joint_limit - epsilon) and (q3_candidate <= joint_limit + epsilon) and (q2_candidate >= -joint_limit - epsilon) and (q2_candidate <= joint_limit + epsilon) and (q1 >= -joint_limit - epsilon) and (q1 <= joint_limit + epsilon):
            return (q1, q2_candidate, q3_candidate, q4_final)
    return (0.0, 0.0, 0.0, 0.0)