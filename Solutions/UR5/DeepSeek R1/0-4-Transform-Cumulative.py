import numpy as np
import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_target = R_z @ R_y @ R_x
    tcp_offset = np.array([0, 0, 0.09465])
    wrist_pos = np.array([x_target, y_target, z_target]) - R_target @ tcp_offset
    wx, wy, wz = wrist_pos
    radius = 0.13585
    adjusted_wx = wx + radius * (wy / math.hypot(wx, wy)) if math.hypot(wx, wy) != 0 else wx
    adjusted_wy = wy - radius * (wx / math.hypot(wx, wy)) if math.hypot(wx, wy) != 0 else wy
    theta1 = math.atan2(-adjusted_wx, adjusted_wy)
    r = math.hypot(wx + 0.13585 * math.sin(theta1), wy - 0.13585 * math.cos(theta1))
    z = wz
    a = 0.425
    b = 0.39225
    d_sq = r ** 2 + z ** 2
    cos_theta3 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3_1 = math.acos(cos_theta3)
    theta3_2 = -theta3_1
    solutions = []
    for theta3 in [theta3_1, theta3_2]:
        k1 = a + b * math.cos(theta3)
        k2 = b * math.sin(theta3)
        theta2 = math.atan2(z, r) - math.atan2(k2, k1)
        r_calc = k1 * math.cos(theta2) - k2 * math.sin(theta2)
        z_calc = k1 * math.sin(theta2) + k2 * math.cos(theta2)
        if math.isclose(r_calc, r, abs_tol=1e-05) and math.isclose(z_calc, z, abs_tol=1e-05):
            solutions.append((theta2, theta3))
    if not solutions:
        theta2, theta3 = (0.0, 0.0)
    else:
        theta2, theta3 = min(solutions, key=lambda x: abs(x[0]) + abs(x[1]))
    R1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R3 = np.array([[math.cos(theta3), 0, math.sin(theta3)], [0, 1, 0], [-math.sin(theta3), 0, math.cos(theta3)]])
    R_total = R1 @ R2 @ R3
    R_remaining = R_total.T @ R_target
    theta4 = math.atan2(R_remaining[0, 2], R_remaining[2, 2])
    R4 = np.array([[math.cos(theta4), 0, math.sin(theta4)], [0, 1, 0], [-math.sin(theta4), 0, math.cos(theta4)]])
    R5 = R4.T @ R_remaining
    theta5 = math.atan2(R5[1, 0], R5[0, 0])
    return (theta1, theta2, theta3, theta4, theta5)