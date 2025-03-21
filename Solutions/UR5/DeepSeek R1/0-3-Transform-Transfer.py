import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    target_roll, target_pitch, target_yaw = r
    q1 = target_yaw
    cos_q1 = np.cos(q1)
    sin_q1 = np.sin(q1)
    x_rot = x_target * cos_q1 + y_target * sin_q1
    y_rot = -x_target * sin_q1 + y_target * cos_q1
    z_rot = z_target
    numerator = x_rot ** 2 + y_rot ** 2 + z_rot ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3_pos = np.arccos(cos_q3)
    q3_neg = -q3_pos
    solutions = []
    for q3 in [q3_pos, q3_neg]:
        A = 0.425 + 0.39225 * np.cos(q3)
        B = 0.39225 * np.sin(q3)
        S_squared = x_rot ** 2 + y_rot ** 2 - 0.10915 ** 2
        S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
        C = z_rot
        numerator_q2 = S * A - C * B
        denominator_q2 = S * B + C * A
        if np.isclose(denominator_q2, 0) and np.isclose(numerator_q2, 0):
            continue
        q2 = np.arctan2(numerator_q2, denominator_q2)
        solutions.append((q2, q3))
    valid_solutions = []
    for q2, q3 in solutions:
        q4 = target_pitch - (q2 + q3)
        q1_norm = (q1 + np.pi) % (2 * np.pi) - np.pi
        q2_norm = (q2 + np.pi) % (2 * np.pi) - np.pi
        q3_norm = (q3 + np.pi) % (2 * np.pi) - np.pi
        q4_norm = (q4 + np.pi) % (2 * np.pi) - np.pi
        if -6.2831853 <= q1_norm <= 6.2831853 and -6.2831853 <= q2_norm <= 6.2831853 and (-6.2831853 <= q3_norm <= 6.2831853) and (-6.2831853 <= q4_norm <= 6.2831853):
            valid_solutions.append((q1_norm, q2_norm, q3_norm, q4_norm))
    if valid_solutions:
        best_error = float('inf')
        best_sol = valid_solutions[0]
        for sol in valid_solutions:
            _, q2_sol, q3_sol, q4_sol = sol
            current_pitch = q2_sol + q3_sol + q4_sol
            error = abs(current_pitch - target_pitch)
            if error < best_error:
                best_error = error
                best_sol = sol
        return best_sol
    else:
        q2, q3 = solutions[0] if solutions else (0.0, 0.0)
        q4 = target_pitch - (q2 + q3)
        q1_norm = (q1 + np.pi) % (2 * np.pi) - np.pi
        q2_norm = (q2 + np.pi) % (2 * np.pi) - np.pi
        q3_norm = (q3 + np.pi) % (2 * np.pi) - np.pi
        q4_norm = (q4 + np.pi) % (2 * np.pi) - np.pi
        return (q1_norm, q2_norm, q3_norm, q4_norm)