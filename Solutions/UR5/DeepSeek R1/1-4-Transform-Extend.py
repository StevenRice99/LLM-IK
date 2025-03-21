import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (np.cos(roll), np.sin(roll))
    cp, sp = (np.cos(pitch), np.sin(pitch))
    cy, sy = (np.cos(yaw), np.sin(yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    theta4 = np.arctan2(R_target[1, 0], R_target[1, 1])
    c4, s4 = (np.cos(theta4), np.sin(theta4))
    c_ts = R_target[0, 0] / c4 if not np.isclose(c4, 0) else 0.0
    s_ts = R_target[0, 2]
    theta_sum = np.arctan2(s_ts, c_ts)
    c_ts_total, s_ts_total = (np.cos(theta_sum), np.sin(theta_sum))
    offset_y_revolute4 = 0.093
    offset_z_tcp = 0.09465
    dx = -offset_y_revolute4 * c_ts_total * s4 + offset_z_tcp * s_ts_total
    dz = offset_y_revolute4 * s_ts_total * s4 + offset_z_tcp * c_ts_total
    X = x_target - dx
    Z = z_target - dz
    L1 = 0.425
    L2 = 0.39225
    d_sq = X ** 2 + Z ** 2
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
        sin_t1 = (C * X - D * Z) / denom
        cos_t1 = (D * X + C * Z) / denom
        if abs(sin_t1) > 1.0 or abs(cos_t1) > 1.0:
            continue
        t1 = np.arctan2(sin_t1, cos_t1)
        t3 = theta_sum - t1 - t2
        t3 = (t3 + np.pi) % (2 * np.pi) - np.pi
        solutions.append((t1, t2, t3))
    best_error = float('inf')
    best_sol = (0.0, 0.0, 0.0)
    for sol in solutions:
        t1, t2, t3 = sol
        if not all((-2 * np.pi <= ang <= 2 * np.pi for ang in sol)):
            continue
        x3 = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
        z3 = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
        error = np.hypot(x3 - X, z3 - Z)
        if error < best_error:
            best_error = error
            best_sol = sol
    theta1, theta2, theta3 = best_sol
    return (theta1, theta2, theta3, theta4)