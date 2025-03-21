import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    target_pos = np.array(p)
    roll, pitch, yaw = r
    cr, cp, cy = (math.cos(roll), math.cos(pitch), math.cos(yaw))
    sr, sp, sy = (math.sin(roll), math.sin(pitch), math.sin(yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    tcp_offset = np.array([0, 0.0823, 0])
    wrist_center = target_pos - R_target @ tcp_offset
    x, y, z = wrist_center
    theta1 = math.atan2(y, x)
    r_xy = math.hypot(x, y)
    x_prime = r_xy
    z_prime = z - 0.13585
    a, b = (0.425, 0.39225)
    d_sq = x_prime ** 2 + z_prime ** 2
    cos_theta3 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = math.acos(cos_theta3)
    theta3_alt = -theta3
    theta2 = math.atan2(x_prime, z_prime) - math.atan2(b * math.sin(theta3), a + b * math.cos(theta3))
    theta2_alt = math.atan2(x_prime, z_prime) - math.atan2(b * math.sin(theta3_alt), a + b * math.cos(theta3_alt))
    pos_error = lambda t2, t3: abs(a * math.sin(t2) + b * math.sin(t2 + t3) - x_prime) + abs(a * math.cos(t2) + b * math.cos(t2 + t3) - z_prime)
    if pos_error(theta2_alt, theta3_alt) < pos_error(theta2, theta3):
        theta2, theta3 = (theta2_alt, theta3_alt)
    R_theta1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R_theta23 = np.array([[math.cos(theta2 + theta3), 0, math.sin(theta2 + theta3)], [0, 1, 0], [-math.sin(theta2 + theta3), 0, math.cos(theta2 + theta3)]])
    R_base = R_theta1 @ R_theta23
    R_wrist = R_base.T @ R_target
    theta5 = math.acos(R_wrist[1, 1])
    if abs(math.sin(theta5)) < 1e-06:
        theta4 = 0.0
        theta6 = math.atan2(-R_wrist[0, 2], R_wrist[2, 2])
    else:
        theta4 = math.atan2(R_wrist[2, 1] / math.sin(theta5), -R_wrist[0, 1] / math.sin(theta5))
        theta6 = math.atan2(R_wrist[1, 2] / math.sin(theta5), -R_wrist[1, 0] / math.sin(theta5))
    angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return tuple(angles.tolist())