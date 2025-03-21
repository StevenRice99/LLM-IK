import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    theta1 = math.atan2(x, z)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    x_rot = x * cos_theta1 + z * sin_theta1
    z_rot = -x * sin_theta1 + z * cos_theta1
    x_target = x_rot
    y_target = y
    z_target = z_rot - 0.39225
    L1 = 0.093
    L2 = 0.09465
    a = 0.0823
    C = y_target - L1
    D_sq = x_target ** 2 + z_target ** 2
    D = math.sqrt(D_sq) if D_sq != 0 else 0.0
    numerator = C ** 2 + D_sq - L2 ** 2 - a ** 2
    denominator = 2 * L2 * a
    if abs(denominator) < 1e-09:
        raise ValueError('Target position is unreachable')
    cos_theta4 = numerator / denominator
    cos_theta4 = max(min(cos_theta4, 1.0), -1.0)
    theta4_options = [math.acos(cos_theta4), -math.acos(cos_theta4)]
    best_solution = None
    min_error = float('inf')
    for theta4 in theta4_options:
        K = L2 + a * math.cos(theta4)
        L = a * math.sin(theta4)
        denominator_theta3 = C ** 2 + D_sq
        if abs(denominator_theta3) < 1e-09:
            continue
        cos_theta3 = (K * C + L * D) / denominator_theta3
        sin_theta3 = (K * D - L * C) / denominator_theta3
        if abs(cos_theta3 ** 2 + sin_theta3 ** 2 - 1) > 1e-06:
            continue
        theta3 = math.atan2(sin_theta3, cos_theta3)
        theta2 = math.atan2(x_target, z_target) if D_sq != 0 else 0.0
        S = math.sqrt(D_sq)
        x_curr = S * math.sin(theta2)
        y_curr = L1 + L2 * math.cos(theta3) + a * math.cos(theta3 + theta4)
        z_curr = S * math.cos(theta2)
        error = (x_curr - x_target) ** 2 + (y_curr - y_target) ** 2 + (z_curr - z_target) ** 2
        if error < min_error:
            min_error = error
            best_solution = (theta2, theta3, theta4)
    if best_solution is None:
        raise ValueError('Target position is unreachable')
    theta2_sub, theta3_sub, theta4_sub = best_solution
    return (theta1, theta2_sub, theta3_sub, theta4_sub)