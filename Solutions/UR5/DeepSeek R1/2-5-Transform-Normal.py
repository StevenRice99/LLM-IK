import numpy as np
from numpy import cos, sin, arctan2, arccos, sqrt, pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    R_desired = euler_to_rotation_matrix(roll, pitch, yaw)
    R_target = R_desired @ rotation_z(-pi / 2)
    y_wrist = y_target - 0.093
    if abs(y_wrist) > 0.0823:
        return (0.0, 0.0, 0.0, 0.0)
    theta3_candidates = np.linspace(-pi, pi, 72)
    for theta3 in theta3_candidates:
        cos_t3 = cos(theta3)
        if abs(cos_t3) < 1e-06:
            continue
        cos_t4 = y_wrist / (0.0823 * cos_t3)
        cos_t4 = np.clip(cos_t4, -1.0, 1.0)
        theta4_options = [arccos(cos_t4), -arccos(cos_t4)]
        for theta4 in theta4_options:
            x_offset = 0.0823 * sin(theta4) * cos(theta3)
            z_offset = 0.0823 * cos(theta4) * cos(theta3)
            x_wrist = x_target - x_offset
            z_wrist = z_target - z_offset
            L1, L2 = (0.39225, 0.09465)
            D_sq = x_wrist ** 2 + z_wrist ** 2
            D = sqrt(D_sq)
            if not abs(L1 - L2) <= D <= L1 + L2:
                continue
            cos_theta2 = (D_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
            theta2 = arccos(cos_theta2)
            theta2_options = [theta2, -theta2]
            for theta2 in theta2_options:
                gamma = arctan2(z_wrist, x_wrist)
                numerator = L2 * sin(theta2)
                denominator = L1 + L2 * cos(theta2)
                delta = arctan2(numerator, denominator)
                theta1 = gamma - delta
                R_combined = rotation_y(theta1) @ rotation_y(theta2) @ rotation_z(theta3) @ rotation_y(theta4)
                if np.allclose(R_combined, R_target, atol=0.001):
                    return (theta1, theta2, theta3, theta4)
    return (0.0, 0.0, 0.0, 0.0)

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    R_y = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    R_z = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    return R_z @ R_y @ R_x

def rotation_y(theta):
    return np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])

def rotation_z(theta):
    return np.array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])