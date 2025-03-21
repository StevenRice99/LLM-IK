import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    target_roll, target_pitch, target_yaw = r
    theta1 = math.atan2(x, z)
    rev2_origin_x = 0.39225 * math.sin(theta1)
    rev2_origin_z = 0.39225 * math.cos(theta1)
    dx = x - rev2_origin_x
    dz = z - rev2_origin_z
    px = dx * math.cos(theta1) - dz * math.sin(theta1)
    pz = dx * math.sin(theta1) + dz * math.cos(theta1)
    py = y
    cr, cp, cy = (math.cos(target_roll), math.cos(target_pitch), math.cos(target_yaw))
    sr, sp, sy = (math.sin(target_roll), math.sin(target_pitch), math.sin(target_yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_revolute1_inv = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    R_target_revolute2 = R_revolute1_inv @ R_target @ R_tcp.T
    solutions = []
    y_offset = py - 0.093
    if abs(y_offset) > 0.0823:
        return (theta1, 0.0, 0.0, 0.0)
    sin_theta3 = y_offset / 0.0823
    theta3_pos = math.asin(sin_theta3)
    theta3_neg = math.pi - theta3_pos
    theta3_candidates = [theta3_pos, theta3_neg]
    for theta3 in theta3_candidates:
        A = 0.0823 * math.cos(theta3)
        B = 0.09465
        C = px
        D = pz - 0.09465
        cos_theta24 = (B * C + A * D) / (A ** 2 + B ** 2)
        sin_theta24 = (A * C - B * D) / (A ** 2 + B ** 2)
        theta24 = math.atan2(sin_theta24, cos_theta24)
        R3 = np.array([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        R24_needed = np.linalg.inv(R3) @ R_target_revolute2
        cos_theta4 = R24_needed[0, 0]
        sin_theta4 = R24_needed[0, 2]
        theta4 = math.atan2(sin_theta4, cos_theta4)
        theta2 = theta24 - theta4
        if not -2 * math.pi <= theta2 <= 2 * math.pi:
            continue
        expected_px = 0.09465 * math.sin(theta2) + 0.0823 * math.sin(theta2 + theta4) * math.cos(theta3)
        expected_py = 0.093 + 0.0823 * math.sin(theta3) * math.sin(theta2 + theta4)
        expected_pz = 0.09465 * math.cos(theta2) + 0.0823 * math.cos(theta2 + theta4) * math.cos(theta3)
        pos_error = math.hypot(px - expected_px, py - expected_py, pz - expected_pz)
        R2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        R4 = np.array([[math.cos(theta4), 0, math.sin(theta4)], [0, 1, 0], [-math.sin(theta4), 0, math.cos(theta4)]])
        R_achieved = R2 @ R3 @ R4 @ R_tcp
        orientation_error = np.linalg.norm(R_achieved - R_target_revolute2)
        total_error = pos_error + orientation_error
        solutions.append((theta2, theta3, theta4, total_error))
    if not solutions:
        return (theta1, 0.0, 0.0, 0.0)
    solutions.sort(key=lambda x: x[3])
    best = solutions[0]
    return (theta1, best[0], best[1], best[2])