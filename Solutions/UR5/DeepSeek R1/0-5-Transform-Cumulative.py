import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    target_pos = np.array(p)
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    d_local = np.array([0, 0.1753, 0.09465])
    WCP = target_pos - R_target @ d_local
    wx, wy, wz = WCP
    A = wy
    B = -wx
    C = 0.13585
    denominator = math.hypot(A, B)
    theta1_candidates = []
    if abs(denominator) < 1e-06:
        theta1_candidates = [0.0]
    else:
        D = C / denominator
        D = np.clip(D, -1.0, 1.0)
        alpha = math.acos(D)
        theta1_1 = math.atan2(B, A) - alpha
        theta1_2 = math.atan2(B, A) + alpha
        theta1_candidates = [theta1_1, theta1_2]
    best_solution = None
    min_error = float('inf')
    for theta1 in theta1_candidates:
        x = wx * math.cos(theta1) + wy * math.sin(theta1)
        y = -wx * math.sin(theta1) + wy * math.cos(theta1) - 0.13585
        z = wz
        if abs(y) > 0.0001:
            continue
        a = 0.425
        b = 0.39225
        d_sq = x ** 2 + z ** 2
        cos_theta3 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
        cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
        theta3 = math.acos(cos_theta3)
        for theta3_val in [theta3, -theta3]:
            k1 = a + b * math.cos(theta3_val)
            k2 = b * math.sin(theta3_val)
            theta2_val = math.atan2(x, z) - math.atan2(k2, k1)
            x_calc = a * math.sin(theta2_val) + b * math.sin(theta2_val + theta3_val)
            z_calc = a * math.cos(theta2_val) + b * math.cos(theta2_val + theta3_val)
            if not (math.isclose(x_calc, x, abs_tol=0.0001) and math.isclose(z_calc, z, abs_tol=0.0001)):
                continue
            R1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
            R2 = np.array([[math.cos(theta2_val), 0, math.sin(theta2_val)], [0, 1, 0], [-math.sin(theta2_val), 0, math.cos(theta2_val)]])
            R3 = np.array([[math.cos(theta3_val), 0, math.sin(theta3_val)], [0, 1, 0], [-math.sin(theta3_val), 0, math.cos(theta3_val)]])
            R_base_to_4 = R1 @ R2 @ R3
            R_4_to_TCP = np.linalg.inv(R_base_to_4) @ R_target
            try:
                theta5 = math.acos(-R_4_to_TCP[1, 2])
            except ValueError:
                continue
            if not np.isclose(math.sin(theta5), 0.0, atol=1e-06):
                s5 = math.sin(theta5)
                theta4 = math.atan2(R_4_to_TCP[2, 2] / s5, R_4_to_TCP[0, 2] / s5)
                theta6 = math.atan2(R_4_to_TCP[1, 0] / s5, -R_4_to_TCP[1, 1] / s5)
            else:
                theta6 = 0.0
                if R_4_to_TCP[1, 2] < 0:
                    theta4 = math.atan2(-R_4_to_TCP[2, 0], -R_4_to_TCP[0, 0])
                else:
                    theta4 = math.atan2(R_4_to_TCP[2, 0], R_4_to_TCP[0, 0])
            current_solution = (theta1 % (2 * math.pi), theta2_val % (2 * math.pi), theta3_val % (2 * math.pi), theta4 % (2 * math.pi), theta5 % (2 * math.pi), theta6 % (2 * math.pi))
            solution_error = sum((abs(j) for j in current_solution))
            if solution_error < min_error:
                min_error = solution_error
                best_solution = current_solution
    return best_solution if best_solution is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)