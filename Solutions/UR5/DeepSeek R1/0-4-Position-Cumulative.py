import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    offset = np.array([0, 0.093, 0.09465])
    numerator = x_target ** 2 + (y_target - offset[1]) ** 2 + z_target ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3_pos = np.arccos(cos_q3)
    q3_neg = -q3_pos
    solutions = []
    for q3 in [q3_pos, q3_neg]:
        A = 0.425 + 0.39225 * np.cos(q3)
        B = 0.39225 * np.sin(q3)
        S_squared = x_target ** 2 + (y_target - offset[1]) ** 2 - 0.10915 ** 2
        S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
        C = z_target
        numerator_q2 = S * A - C * B
        denominator_q2 = S * B + C * A
        q2 = np.arctan2(numerator_q2, denominator_q2)
        phi = np.arctan2(0.10915, S) if S != 0 else np.pi / 2
        q1 = np.arctan2(y_target - offset[1], x_target) - phi
        q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
        R1 = np.array([[np.cos(q1), -np.sin(q1), 0], [np.sin(q1), np.cos(q1), 0], [0, 0, 1]])
        R2 = np.array([[np.cos(q2), 0, np.sin(q2)], [0, 1, 0], [-np.sin(q2), 0, np.cos(q2)]])
        R3 = np.array([[np.cos(q3), 0, np.sin(q3)], [0, 1, 0], [-np.sin(q3), 0, np.cos(q3)]])
        P4 = np.array([0, 0.13585, 0])
        P4 = P4 + R1 @ np.array([0, -0.1197, 0.425])
        P4 = P4 + R1 @ R2 @ np.array([0, 0, 0.39225])
        D = np.array([x_target, y_target, z_target]) - P4
        R_total = R1 @ R2 @ R3
        D_local = np.linalg.inv(R_total) @ D
        target_xz = np.array([D_local[0], D_local[2]])
        norm_xz = np.linalg.norm(target_xz)
        if not np.isclose(norm_xz, np.linalg.norm(offset[[0, 2]]), atol=0.001):
            continue
        q4 = np.arctan2(D_local[0] / 0.09465, D_local[2] / 0.09465)
        if not np.isclose(D_local[1], 0.093, atol=0.001):
            continue
        solutions.append((q1, q2, q3, q4, P4))
    min_error = float('inf')
    best_sol = None
    for sol in solutions:
        q1, q2, q3, q4, P4 = sol
        R4 = np.array([[np.cos(q4), 0, np.sin(q4)], [0, 1, 0], [-np.sin(q4), 0, np.cos(q4)]])
        rotated_offset = R1 @ R2 @ R3 @ R4 @ offset
        tcp = P4 + rotated_offset
        error = np.linalg.norm(tcp - np.array([x_target, y_target, z_target]))
        if error < min_error:
            min_error = error
            best_sol = (q1, q2, q3, q4)
    if best_sol is None:
        return inverse_kinematics_fallback(p)
    q1, q2, q3, q4 = best_sol
    q5 = 0.0
    return (q1, q2, q3, q4, q5)

def inverse_kinematics_fallback(p):
    x, y, z = p
    offset_y = 0.093
    numerator = x ** 2 + (y - offset_y) ** 2 + z ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    S_squared = x ** 2 + (y - offset_y) ** 2 - 0.10915 ** 2
    S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = z
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = np.arctan2(numerator_q2, denominator_q2)
    phi = np.arctan2(0.10915, S) if S != 0 else np.pi / 2
    q1 = np.arctan2(y - offset_y, x) - phi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    q4 = 0.0
    q5 = 0.0
    return (q1, q2, q3, q4, q5)