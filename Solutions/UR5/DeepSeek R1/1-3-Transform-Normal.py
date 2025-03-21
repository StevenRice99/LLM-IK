import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (np.cos(roll), np.sin(roll))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cy, sy = (np.cos(yaw), np.sin(yaw))
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    cos_theta = R[0, 0]
    sin_theta = R[0, 2]
    target_sum = np.arctan2(sin_theta, cos_theta)
    L1 = 0.425
    L2 = 0.39225
    d_sq = x_target ** 2 + z_target ** 2
    cos_theta2 = (d_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    theta2_alt = -theta2
    solutions = []
    for t2 in [theta2, theta2_alt]:
        C = L1 + L2 * np.cos(t2)
        D = L2 * np.sin(t2)
        denom = C ** 2 + D ** 2
        if denom < 1e-06:
            continue
        sin_t1 = (C * x_target - D * z_target) / denom
        cos_t1 = (D * x_target + C * z_target) / denom
        if abs(sin_t1) > 1.0 or abs(cos_t1) > 1.0:
            continue
        t1 = np.arctan2(sin_t1, cos_t1)
        t3_base = (target_sum - t1 - t2) % (2 * np.pi)
        t3_candidates = [t3_base - 2 * np.pi, t3_base, t3_base + 2 * np.pi]
        for t3 in t3_candidates:
            if -2 * np.pi <= t3 <= 2 * np.pi:
                solutions.append((t1, t2, t3))
    best_error = float('inf')
    best_sol = (0.0, 0.0, 0.0)
    for sol in solutions:
        t1, t2, t3 = sol
        if not (-2 * np.pi <= t1 <= 2 * np.pi and -2 * np.pi <= t2 <= 2 * np.pi and (-2 * np.pi <= t3 <= 2 * np.pi)):
            continue
        x = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
        z = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
        pos_error = np.hypot(x - x_target, z - z_target)
        orient_sum = (t1 + t2 + t3) % (2 * np.pi)
        target_orient = target_sum % (2 * np.pi)
        orient_error = min(abs(orient_sum - target_orient), 2 * np.pi - abs(orient_sum - target_orient))
        total_error = pos_error + orient_error
        if total_error < best_error:
            best_error = total_error
            best_sol = sol
    return best_sol