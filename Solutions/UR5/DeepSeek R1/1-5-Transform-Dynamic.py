import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    target_roll, target_pitch, target_yaw = r
    cr, cp, cy = (math.cos(target_roll), math.cos(target_pitch), math.cos(target_yaw))
    sr, sp, sy = (math.sin(target_roll), math.sin(target_pitch), math.sin(target_yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    tcp_offset = np.array([0, 0.0823, 0.09465])
    wrist_center = np.array(p) - R_target @ tcp_offset
    x_wc, _, z_wc = wrist_center
    y_wc = -0.0267
    a1, a2 = (0.425, 0.39225)
    dx, dz = (x_wc, z_wc)
    D_sq = dx ** 2 + dz ** 2
    D = math.sqrt(D_sq)
    cos_theta2 = (D_sq - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(dx, dz) - math.atan2(a2 * math.sin(theta2), a1 + a2 * math.cos(theta2))
    R1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    R2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R_joints_12 = R1 @ R2
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_remaining = R_joints_12.T @ R_target @ R_tcp.T

    def yzy_decomposition(R):
        theta4 = math.atan2(math.sqrt(R[0, 1] ** 2 + R[2, 1] ** 2), R[1, 1])
        if np.isclose(math.sin(theta4), 0):
            theta3 = 0.0
            theta5 = math.atan2(-R[2, 0], R[0, 0])
        else:
            theta3 = math.atan2(R[2, 1] / math.sin(theta4), R[0, 1] / math.sin(theta4))
            theta5 = math.atan2(R[1, 2] / math.sin(theta4), -R[1, 0] / math.sin(theta4))
        return (theta3, theta4, theta5)
    theta3, theta4, theta5 = yzy_decomposition(R_remaining)
    return (theta1, theta2, theta3, theta4, theta5)