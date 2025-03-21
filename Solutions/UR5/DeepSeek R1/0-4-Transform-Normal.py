import numpy as np
from math import atan2, sqrt, acos, sin, cos

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = atan2(y, x)
    r_proj = sqrt(x ** 2 + y ** 2)
    a2 = 0.425
    a3 = 0.39225
    a4 = sqrt(0.093 ** 2 + 0.09465 ** 2)
    dx = r_proj
    dy = -0.13585
    dz = z
    d = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if not abs(a2 - (a3 + a4)) <= d <= a2 + a3 + a4:
        raise ValueError('Target position is unreachable')
    cos_theta3 = (d ** 2 - a2 ** 2 - (a3 + a4) ** 2) / (2 * a2 * (a3 + a4))
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -acos(cos_theta3)
    gamma = atan2(dz, dx)
    delta = acos((a2 ** 2 + d ** 2 - (a3 + a4) ** 2) / (2 * a2 * d))
    theta2 = gamma - delta
    theta4 = -(theta2 + theta3)
    R1 = np.array([[cos(theta1), -sin(theta1), 0], [sin(theta1), cos(theta1), 0], [0, 0, 1]])
    R2 = np.array([[cos(theta2), 0, sin(theta2)], [0, 1, 0], [-sin(theta2), 0, cos(theta2)]])
    R3 = np.array([[cos(theta3), 0, sin(theta3)], [0, 1, 0], [-sin(theta3), 0, cos(theta3)]])
    R4 = np.array([[cos(theta4), 0, sin(theta4)], [0, 1, 0], [-sin(theta4), 0, cos(theta4)]])
    R_combined = R1 @ R2 @ R3 @ R4
    cr, sr = (cos(roll), sin(roll))
    cp, sp = (cos(pitch), sin(pitch))
    cy, sy = (cos(yaw), sin(yaw))
    R_target = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    R_diff = R_combined.T @ R_target
    theta5 = atan2(R_diff[1, 0], R_diff[0, 0])
    return (theta1, theta2, theta3, theta4, theta5)