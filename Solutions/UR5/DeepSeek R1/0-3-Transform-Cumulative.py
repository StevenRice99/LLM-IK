import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    offset_tcp = np.array([0, 0.093, 0])
    offset_world = R_target.dot(offset_tcp)
    P4 = np.array([x_target, y_target, z_target]) - offset_world
    x_p4, y_p4, z_p4 = P4
    theta1 = math.atan2(-x_p4, y_p4)
    revolute2_x = 0.13585 * math.sin(theta1)
    revolute2_y = 0.13585 * math.cos(theta1)
    revolute2_z = 0.0
    x_rel = x_p4 - revolute2_x
    y_rel = y_p4 - revolute2_y
    z_rel = z_p4 - revolute2_z
    x_plane = x_rel * math.cos(theta1) + y_rel * math.sin(theta1)
    z_plane = z_rel
    L1 = 0.425
    L2 = 0.39225
    d_sq = x_plane ** 2 + z_plane ** 2
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
        sin_t1 = (C * x_plane - D * z_plane) / denom
        cos_t1 = (D * x_plane + C * z_plane) / denom
        if abs(sin_t1) > 1.0 or abs(cos_t1) > 1.0:
            continue
        t1 = np.arctan2(sin_t1, cos_t1)
        solutions.append((t1, t2))
    best_error = float('inf')
    best_theta2, best_theta3 = (0.0, 0.0)
    for t1, t2 in solutions:
        R1_j = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
        R2_j = np.array([[math.cos(t1), 0, math.sin(t1)], [0, 1, 0], [-math.sin(t1), 0, math.cos(t1)]])
        R3_j = np.array([[math.cos(t2), 0, math.sin(t2)], [0, 1, 0], [-math.sin(t2), 0, math.cos(t2)]])
        R_123 = R1_j.dot(R2_j).dot(R3_j)
        R_remaining = R_123.T.dot(R_target)
        theta4 = math.atan2(R_remaining[0, 2], R_remaining[2, 2])
        R4_j = np.array([[math.cos(theta4), 0, math.sin(theta4)], [0, 1, 0], [-math.sin(theta4), 0, math.cos(theta4)]])
        R_total = R_123.dot(R4_j)
        orient_error = np.linalg.norm(R_total - R_target)
        if orient_error < best_error:
            best_error = orient_error
            best_theta2, best_theta3 = (t1, t2)
    R1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(best_theta2), 0, math.sin(best_theta2)], [0, 1, 0], [-math.sin(best_theta2), 0, math.cos(best_theta2)]])
    R3 = np.array([[math.cos(best_theta3), 0, math.sin(best_theta3)], [0, 1, 0], [-math.sin(best_theta3), 0, math.cos(best_theta3)]])
    R_123 = R1.dot(R2).dot(R3)
    R_remaining = R_123.T.dot(R_target)
    theta4 = math.atan2(R_remaining[0, 2], R_remaining[2, 2])
    return (theta1, best_theta2, best_theta3, theta4)