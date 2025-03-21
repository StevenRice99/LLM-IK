import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    theta1, theta2 = inverse_kinematics_existing1((x_target, 0, z_target))
    pos_revolute3_x = 0.425 * math.sin(theta1) + 0.39225 * math.sin(theta1 + theta2)
    pos_revolute3_z = 0.425 * math.cos(theta1) + 0.39225 * math.cos(theta1 + theta2)
    pos_revolute3_y = -0.1197
    dx = x_target - pos_revolute3_x
    dy = y_target - pos_revolute3_y
    dz = z_target - pos_revolute3_z
    total_angle = theta1 + theta2
    cos_ta = math.cos(total_angle)
    sin_ta = math.sin(total_angle)
    dx_local = dx * cos_ta + dz * sin_ta
    dz_local = -dx * sin_ta + dz * cos_ta
    dy_local = dy
    theta3, theta4 = solve_last_two_joints(dx_local, dy_local, dz_local)
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)

def inverse_kinematics_existing1(p):
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d = math.hypot(x, z)
    numerator = d ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_theta2 = numerator / denominator
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    cross_product = x * (L1 + L2 * math.cos(theta2)) - z * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x, z)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    return (theta1, theta2)

def solve_last_two_joints(dx_local, dy_local, dz_local):
    L4 = 0.093
    a = 0.0823
    L5 = 0.09465
    y_eq_rhs = (dy_local - L4) / a
    y_eq_rhs = max(min(y_eq_rhs, 1.0), -1.0)
    if abs(y_eq_rhs) > 1.0:
        raise ValueError('Target unreachable for last three joints')
    theta4_pos = math.acos(y_eq_rhs)
    theta4_neg = -theta4_pos
    theta4_options = [theta4_pos, theta4_neg]
    best_error = float('inf')
    best_solution = None
    for theta4 in theta4_options:
        sin_theta4 = math.sin(theta4)
        cos_theta4 = math.cos(theta4)
        A = -a * sin_theta4
        B = L5
        C = dx_local
        D = a * sin_theta4
        E = L5
        F = dz_local
        denominator = B * E - A * D
        if abs(denominator) < 1e-09:
            continue
        sin_theta3 = (C * E - A * F) / denominator
        cos_theta3 = (B * F - C * D) / denominator
        norm = math.hypot(sin_theta3, cos_theta3)
        if abs(norm - 1.0) > 1e-06:
            continue
        theta3 = math.atan2(sin_theta3, cos_theta3)
        x_achieved = B * sin_theta3 + A * cos_theta3
        z_achieved = E * cos_theta3 + D * sin_theta3
        error = (x_achieved - dx_local) ** 2 + (z_achieved - dz_local) ** 2
        if error < best_error:
            best_error = error
            best_solution = (theta3, theta4)
    if best_solution is None:
        raise ValueError('Target unreachable for last three joints')
    return best_solution