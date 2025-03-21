import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    target_roll, target_pitch, target_yaw = r
    cr, sr = (math.cos(target_roll), math.sin(target_roll))
    cp, sp = (math.cos(target_pitch), math.sin(target_pitch))
    cy, sy = (math.cos(target_yaw), math.sin(target_yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_target_adj = R_target @ R_tcp.T
    L4 = 0.093
    L5 = 0.09465
    L_tcp = 0.0823
    tcp_offset_local = np.array([0, L_tcp, L5])
    tcp_offset_global = R_target_adj @ tcp_offset_local
    wx = x_tcp - tcp_offset_global[0]
    wy = -L4
    wz = z_tcp - tcp_offset_global[2]
    L1_y = -0.1197
    L1_z = 0.425
    L2_z = 0.39225
    x = math.sqrt(wx ** 2 + wz ** 2)
    y = wy - L1_y
    z = wz
    D_sq = x ** 2 + y ** 2 + (z - L1_z) ** 2
    L_eff = math.sqrt(L2_z ** 2 + L4 ** 2)
    cos_theta3 = (D_sq - L1_z ** 2 - L_eff ** 2) / (2 * L1_z * L_eff)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = math.acos(cos_theta3)
    solutions = []
    for sign in [1, -1]:
        theta3_candidate = sign * theta3
        A = L1_z + L_eff * math.cos(theta3_candidate)
        B = L_eff * math.sin(theta3_candidate)
        theta1 = math.atan2(x, z) - math.atan2(B, A)
        theta2_candidate = math.atan2(y, math.sqrt(x ** 2 + (z - A) ** 2))
        x_check = L1_z * math.sin(theta1) + L_eff * math.sin(theta1 + theta3_candidate)
        z_check = L1_z * math.cos(theta1) + L_eff * math.cos(theta1 + theta3_candidate)
        y_check = L1_y + L4 * math.sin(theta2_candidate)
        position_error = math.hypot(x_check - x, z_check - z) + abs(y_check - y)
        solutions.append((theta1, theta2_candidate, theta3_candidate, position_error))
    solutions.sort(key=lambda x: x[3])
    theta1, theta2, theta3, _ = solutions[0]
    R_sum = np.array([[math.cos(theta1 + theta3), 0, math.sin(theta1 + theta3)], [0, 1, 0], [-math.sin(theta1 + theta3), 0, math.cos(theta1 + theta3)]])
    R_remaining = R_sum.T @ R_target_adj
    theta5 = math.atan2(-R_remaining[2, 0], R_remaining[2, 2])
    theta4 = math.atan2(-R_remaining[0, 1], R_remaining[1, 1])
    return (theta1, theta2, theta3, theta4, theta5)