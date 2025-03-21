import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    d1 = 0.13585
    a2 = 0.425
    d2 = 0.1197
    a3 = 0.39225
    d4 = 0.093
    d5 = 0.09465
    L2 = math.sqrt(a2 ** 2 + d2 ** 2)
    target_z_adj = z_target - d1
    cos_theta2 = target_z_adj / L2
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    A = a2 * sin_theta2
    B = d2
    denom = A ** 2 + B ** 2
    C = (A * x_target + B * y_target) / denom
    S = (-B * x_target + A * y_target) / denom
    theta1 = math.atan2(S, C)
    x3 = A * math.cos(theta1) - B * math.sin(theta1)
    y3 = A * math.sin(theta1) + B * math.cos(theta1)
    z3 = a2 * cos_theta2 + d1
    dx = x_target - x3
    dy = y_target - y3
    dz = z_target - z3
    dx_rot_z = dx * math.cos(theta1) + dy * math.sin(theta1)
    dy_rot_z = -dx * math.sin(theta1) + dy * math.cos(theta1)
    dz_rot_z = dz
    dx_local = dx_rot_z * math.cos(theta2) + dz_rot_z * math.sin(theta2)
    dz_local = -dx_rot_z * math.sin(theta2) + dz_rot_z * math.cos(theta2)
    dy_local = dy_rot_z - d4
    L_remaining = math.sqrt(a3 ** 2 + d5 ** 2)
    target_dist = math.sqrt(dx_local ** 2 + dz_local ** 2)
    if abs(target_dist) < 1e-06:
        return (theta1, theta2, 0.0, 0.0, 0.0)
    cos_theta4 = (target_dist ** 2 - a3 ** 2 - d5 ** 2) / (2 * a3 * d5)
    cos_theta4 = max(min(cos_theta4, 1.0), -1.0)
    theta4 = math.acos(cos_theta4)
    sin_theta4 = math.sin(theta4)
    A_ik = a3 + d5 * cos_theta4
    B_ik = d5 * sin_theta4
    denom_theta3 = A_ik ** 2 + B_ik ** 2
    sin_theta3 = (A_ik * dx_local - B_ik * dz_local) / denom_theta3
    cos_theta3 = (B_ik * dx_local + A_ik * dz_local) / denom_theta3
    theta3 = math.atan2(sin_theta3, cos_theta3)
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)