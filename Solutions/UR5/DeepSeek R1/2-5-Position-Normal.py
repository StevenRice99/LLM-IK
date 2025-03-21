def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    import numpy as np
    x_target, y_target, z_target = p
    q4 = 0.0
    numerator = y_target - 0.093
    denominator = 0.0823
    if abs(numerator) > abs(denominator):
        return (0.0, 0.0, 0.0, 0.0)
    cos_q3 = numerator / denominator
    q3_0 = np.arccos(cos_q3)
    q3_1 = -q3_0
    q3_candidates = [q3_0, q3_1]
    for q3 in q3_candidates:
        x_offset = -0.0823 * np.sin(q3)
        x_remaining = x_target - x_offset
        L1 = 0.39225
        L2 = 0.39225 + 0.09465
        D_sq = x_remaining ** 2 + z_target ** 2
        D = np.sqrt(D_sq)
        if D > L1 + L2 or D < abs(L1 - L2):
            continue
        cos_q2 = (D_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        if cos_q2 < -1 or cos_q2 > 1:
            continue
        q2 = np.arccos(cos_q2)
        q2_candidates = [q2, -q2]
        for q2 in q2_candidates:
            gamma = np.arctan2(z_target, x_remaining)
            beta = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
            q1 = gamma - beta
            x_check = L1 * np.sin(q1) + L2 * np.sin(q1 + q2) - 0.0823 * np.sin(q3)
            z_check = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
            y_check = 0.093 + 0.0823 * np.cos(q3)
            if np.isclose(x_check, x_target, atol=0.0001) and np.isclose(z_check, z_target, atol=0.0001) and np.isclose(y_check, y_target, atol=0.0001):
                return (q1, q2, q3, q4)
    return (0.0, 0.0, 0.0, 0.0)