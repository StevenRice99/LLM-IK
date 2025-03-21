import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    roll_target, pitch_target, yaw_target = r
    L2_y = 0.13585
    D = math.hypot(x_target, y_target)
    theta1_candidates = []
    if D < 1e-06:
        theta1_candidates = [math.atan2(-0.0, 1.0)]
    else:
        cos_alpha = L2_y / D
        if abs(cos_alpha) <= 1.0:
            alpha = math.acos(cos_alpha)
            base_angle = math.atan2(-x_target, y_target)
            theta1_candidates = [base_angle + alpha, base_angle - alpha]
    best_error = float('inf')
    best_solution = (0.0, 0.0, 0.0, 0.0)
    for theta1 in theta1_candidates:
        cos_t1 = math.cos(theta1)
        sin_t1 = math.sin(theta1)
        x_rot = x_target * cos_t1 + y_target * sin_t1
        y_rot = -x_target * sin_t1 + y_target * cos_t1 - L2_y
        z_rot = z_target
        if abs(y_rot) > 0.0001:
            continue
        L1 = 0.425
        L2 = 0.39225
        x_arm = x_rot
        z_arm = z_rot
        adjusted_yaw = (yaw_target - theta1) % (2 * math.pi)
        cr, sr = (math.cos(roll_target), math.sin(roll_target))
        cp, sp = (math.cos(pitch_target), math.sin(pitch_target))
        cy, sy = (math.cos(adjusted_yaw), math.sin(adjusted_yaw))
        R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
        target_sum = math.atan2(R[0, 2], R[0, 0])
        d_sq = x_arm ** 2 + z_arm ** 2
        cos_theta2 = (d_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
        theta2 = math.acos(cos_theta2)
        theta2_alt = -theta2
        solutions = []
        for t2 in [theta2, theta2_alt]:
            C = L1 + L2 * math.cos(t2)
            D_val = L2 * math.sin(t2)
            denom = C ** 2 + D_val ** 2
            if denom < 1e-06:
                continue
            sin_t1_arm = (C * x_arm - D_val * z_arm) / denom
            cos_t1_arm = (D_val * x_arm + C * z_arm) / denom
            if abs(sin_t1_arm) > 1.0 or abs(cos_t1_arm) > 1.0:
                continue
            t1_arm = math.atan2(sin_t1_arm, cos_t1_arm)
            t3 = (target_sum - t1_arm - t2) % (2 * math.pi)
            solutions.extend([(t1_arm, t2, t3), (t1_arm, t2, t3 - 2 * math.pi), (t1_arm, t2, t3 + 2 * math.pi)])
        for sol in solutions:
            t1_arm, t2, t3 = sol
            if not all((-6.3 < angle < 6.3 for angle in (t1_arm, t2, t3))):
                continue
            x_actual = L1 * math.sin(t1_arm) + L2 * math.sin(t1_arm + t2)
            z_actual = L1 * math.cos(t1_arm) + L2 * math.cos(t1_arm + t2)
            pos_error = math.hypot(x_actual - x_arm, z_actual - z_arm)
            orient_sum = (t1_arm + t2 + t3) % (2 * math.pi)
            orient_error = min(abs(orient_sum - target_sum), 2 * math.pi - abs(orient_sum - target_sum))
            total_error = pos_error + 0.5 * orient_error
            if total_error < best_error:
                best_error = total_error
                best_solution = (theta1, t1_arm, t2, t3)
    return best_solution