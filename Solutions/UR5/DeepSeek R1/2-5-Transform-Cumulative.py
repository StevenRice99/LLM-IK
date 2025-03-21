import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    target_roll, target_pitch, target_yaw = r
    cr = math.cos(target_roll)
    sr = math.sin(target_roll)
    cp = math.cos(target_pitch)
    sp = math.sin(target_pitch)
    cy = math.cos(target_yaw)
    sy = math.sin(target_yaw)
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_tcp_inv = R_tcp.T
    a = 0.39225
    d = 0.09465
    l = 0.0823
    solutions = []
    for sign in [1, -1]:
        cos_theta4 = (z_target - a * math.cos(target_pitch) - d) / l
        cos_theta4 = np.clip(cos_theta4, -1.0, 1.0)
        theta4 = sign * math.acos(cos_theta4)
        sin_theta1 = (x_target - l * math.sin(theta4)) / a
        sin_theta1 = np.clip(sin_theta1, -1.0, 1.0)
        theta1 = math.asin(sin_theta1)
        current_z = a * math.cos(theta1) + d + l * math.cos(theta4)
        if abs(current_z - z_target) > 1e-06:
            continue
        R1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
        R4 = np.array([[math.cos(theta4), 0, math.sin(theta4)], [0, 1, 0], [-math.sin(theta4), 0, math.cos(theta4)]])
        R_remaining = np.linalg.inv(R1 @ R4) @ R_target @ R_tcp_inv
        theta2 = math.atan2(R_remaining[2, 1], R_remaining[2, 2])
        theta3 = math.atan2(-R_remaining[0, 1], R_remaining[1, 1])
        R2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
        R3 = np.array([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
        R_total = R1 @ R2 @ R3 @ R4 @ R_tcp
        error = np.linalg.norm(R_total - R_target)
        solutions.append((theta1, theta2, theta3, theta4, error))
    if not solutions:
        return (0.0, 0.0, 0.0, 0.0)
    solutions.sort(key=lambda x: x[4])
    best = solutions[0]
    return (best[0], best[1], best[2], best[3])