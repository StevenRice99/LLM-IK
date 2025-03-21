import numpy as np
import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    R_roll = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R_pitch = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R_yaw = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_target = R_yaw @ R_pitch @ R_roll
    r21 = R_target[1, 0]
    r22 = R_target[1, 1]
    theta4 = math.atan2(r21, r22)
    r13 = R_target[0, 2]
    r33 = R_target[2, 2]
    theta_sum = math.atan2(r13, r33)
    x_adj = x_target - 0.09465 * math.sin(theta_sum)
    z_adj = z_target - 0.09465 * math.cos(theta_sum)
    L1 = 0.425
    L2 = 0.39225
    D_sq = x_adj ** 2 + z_adj ** 2
    D = math.sqrt(D_sq)
    if not abs(L1 - L2) <= D <= L1 + L2:
        raise ValueError('Target position is unreachable')
    cos_theta2 = (D_sq - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -math.acos(cos_theta2)
    alpha = math.atan2(z_adj, x_adj)
    beta = math.atan2(L2 * math.sin(-theta2), L1 + L2 * math.cos(-theta2))
    theta1 = alpha - beta
    theta3 = theta_sum - theta1 - theta2
    return (theta1, theta2, theta3, theta4)