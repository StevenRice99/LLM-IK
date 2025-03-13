import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    y_adjusted = y_target + 0.0267
    cos_q1 = y_adjusted / 0.13585
    cos_q1 = np.clip(cos_q1, -1.0, 1.0)
    q1_pos = np.arccos(cos_q1)
    q1_neg = -q1_pos
    q1_candidates = [q1_pos, q1_neg, q1_pos - 2 * np.pi, q1_neg + 2 * np.pi]
    valid_solutions = []
    for q1 in q1_candidates:
        x_revolute2 = 0.13585 * np.sin(q1)
        y_revolute2 = 0.13585 * np.cos(q1)
        x_prime = x_target - x_revolute2
        y_prime = y_target - y_revolute2
        dx = x_prime * np.cos(q1) + y_prime * np.sin(q1)
        dz = z_target
        L1 = 0.425
        L2_eff = 0.39225 + 0.093
        target_x = dx
        target_z = dz - 0.093
        D_sq = target_x ** 2 + target_z ** 2
        if D_sq < 1e-07:
            continue
        cos_q3 = (D_sq - L1 ** 2 - L2_eff ** 2) / (2 * L1 * L2_eff)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = np.arccos(cos_q3)
        q3_neg = -q3
        for q3_sol in [q3, q3_neg]:
            denominator = L1 + L2_eff * np.cos(q3_sol)
            numerator = L2_eff * np.sin(q3_sol)
            angle_offset = np.arctan2(numerator, denominator)
            q2 = np.arctan2(target_x, target_z) - angle_offset
            x_sol = L1 * np.sin(q2) + L2_eff * np.sin(q2 + q3_sol)
            z_sol = L1 * np.cos(q2) + L2_eff * np.cos(q2 + q3_sol) + 0.093
            if np.isclose(x_sol, dx, atol=1e-05) and np.isclose(z_sol, dz, atol=1e-05):
                q4 = -(q2 + q3_sol)
                valid_solutions.append((q1, q2, q3_sol, q4))
    if valid_solutions:
        q1, q2, q3, q4 = valid_solutions[0]
        return (q1 % (2 * np.pi), q2 % (2 * np.pi), q3 % (2 * np.pi), q4 % (2 * np.pi))
    else:
        return (0.0, 0.0, 0.0, 0.0)