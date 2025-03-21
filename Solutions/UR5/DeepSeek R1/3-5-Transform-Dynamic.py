import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    px, py, pz = p
    target_roll, target_pitch, target_yaw = r
    y_offset = py - 0.093
    if abs(y_offset) > 0.0823:
        raise ValueError('Target position is unreachable based on y-coordinate.')
    cos_theta2 = y_offset / 0.0823
    theta2_pos = math.acos(cos_theta2)
    theta2_neg = -theta2_pos
    possible_theta2 = [theta2_pos, theta2_neg]
    solutions = []
    cr, cp, cy = (math.cos(target_roll), math.cos(target_pitch), math.cos(target_yaw))
    sr, sp, sy = (math.sin(target_roll), math.sin(target_pitch), math.sin(target_yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_tcp_inv = R_tcp.T
    for theta2 in possible_theta2:
        sin_theta2 = math.sin(theta2)
        A = -0.0823 * sin_theta2
        B = 0.09465
        C = 0.0823 * sin_theta2
        D = 0.09465
        det = B * D - A * C
        if abs(det) < 1e-06:
            continue
        sin_theta1 = (D * px - A * pz) / det
        cos_theta1 = (-C * px + B * pz) / det
        theta1 = math.atan2(sin_theta1, cos_theta1)
        R1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
        R2 = np.array([[math.cos(theta2), -math.sin(theta2), 0], [math.sin(theta2), math.cos(theta2), 0], [0, 0, 1]])
        R_desired = R_target @ R_tcp_inv
        R_total = R1 @ R2
        R3 = np.linalg.inv(R_total) @ R_desired
        cos_theta3 = R3[0, 0]
        sin_theta3 = R3[0, 2]
        theta3 = math.atan2(sin_theta3, cos_theta3)
        R3_actual = np.array([[math.cos(theta3), 0, math.sin(theta3)], [0, 1, 0], [-math.sin(theta3), 0, math.cos(theta3)]])
        R_achieved = R_total @ R3_actual @ R_tcp
        orientation_error = np.linalg.norm(R_achieved - R_target)
        solutions.append((theta1, theta2, theta3, orientation_error))
    if not solutions:
        return (0.0, 0.0, 0.0)
    solutions.sort(key=lambda x: x[3])
    best_solution = solutions[0][:3]
    return best_solution