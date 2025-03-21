import numpy as np
from numpy import sin, cos, arctan2, sqrt, pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    cr, sr = (cos(roll), sin(roll))
    cp, sp = (cos(pitch), sin(pitch))
    cy, sy = (cos(yaw), sin(yaw))
    R_desired = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    d_tcp = np.array([0, 0.0823, 0])
    d_tcp_world = R_desired @ R_tcp.T @ d_tcp
    WC = np.array([x, y, z]) - d_tcp_world
    WC_x, WC_y, WC_z = WC
    A = 0.13585
    K_sq = WC_x ** 2 + WC_y ** 2 - A ** 2
    K = sqrt(K_sq) if K_sq >= 0 else 0.0
    theta1_options = []
    for K_sign in [1, -1]:
        K_val = K * K_sign
        denom = A ** 2 + K_val ** 2
        if denom == 0:
            continue
        sin_theta1 = (A * WC_x + K_val * WC_y) / denom
        cos_theta1 = (K_val * WC_x - A * WC_y) / denom
        theta1 = arctan2(sin_theta1, cos_theta1)
        theta1_options.append(theta1)
    B, C = (0.425, 0.39225)
    solutions = []
    for theta1 in theta1_options:
        D = WC_x * cos(theta1) + WC_y * sin(theta1) - A
        for elbow_sign in [1, -1]:
            cos_theta3 = (D ** 2 + WC_z ** 2 - B ** 2 - C ** 2) / (2 * B * C)
            cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
            sin_theta3 = elbow_sign * sqrt(1 - cos_theta3 ** 2)
            theta3 = arctan2(sin_theta3, cos_theta3)
            A1 = B + C * cos_theta3
            A2 = C * sin_theta3
            denom = A1 ** 2 + A2 ** 2
            if denom == 0:
                continue
            sin_theta2 = (A1 * WC_z - A2 * D) / denom
            cos_theta2 = (A1 * D + A2 * WC_z) / denom
            theta2 = arctan2(sin_theta2, cos_theta2)
            R1 = np.array([[cos(theta1), -sin(theta1), 0], [sin(theta1), cos(theta1), 0], [0, 0, 1]])
            R2 = np.array([[cos(theta2), 0, sin(theta2)], [0, 1, 0], [-sin(theta2), 0, cos(theta2)]])
            R3 = np.array([[cos(theta3), 0, sin(theta3)], [0, 1, 0], [-sin(theta3), 0, cos(theta3)]])
            R_base_to_wrist = R1 @ R2 @ R3
            R_wrist = R_base_to_wrist.T @ R_desired @ R_tcp.T
            r13 = R_wrist[0, 2]
            r23 = R_wrist[1, 2]
            r33 = R_wrist[2, 2]
            r12 = R_wrist[0, 1]
            r32 = R_wrist[2, 1]
            theta5 = arctan2(sqrt(r13 ** 2 + r33 ** 2), -r23)
            if abs(sin(theta5)) < 1e-06:
                theta4 = 0.0
                theta6 = arctan2(-r32, r12)
            else:
                theta4 = arctan2(r33, r13)
                theta6 = arctan2(r32, -r12)
            solutions.append((theta1, theta2, theta3, theta4, theta5, theta6))
    return min(solutions, key=lambda s: abs(s[3] + s[5])) if solutions else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)