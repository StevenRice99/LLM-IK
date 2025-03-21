import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    y_adjusted = y_target + 0.0267
    cos_theta4 = y_adjusted / 0.0823
    if cos_theta4 < -1 or cos_theta4 > 1:
        raise ValueError('Target y-coordinate is out of reachable range')
    theta4 = math.acos(cos_theta4)
    theta4_solutions = [theta4, -theta4]
    for theta4 in theta4_solutions:
        R = math.hypot(0.0823 * math.sin(theta4), 0.09465)
        phi = math.atan2(0.0823 * math.sin(theta4), 0.09465)
        L1_plus_L2 = 0.425 + 0.39225
        K_numerator = x_target ** 2 + z_target ** 2 - R ** 2 - L1_plus_L2 ** 2
        K = K_numerator / (2 * R * L1_plus_L2)
        if K < -1 or K > 1:
            continue
        C = math.acos(-K)
        alpha = math.atan2(x_target, z_target)
        theta_total1 = C - alpha - phi
        theta_total2 = -C - alpha - phi
        for theta_total in [theta_total1, theta_total2]:
            D_x = R * math.sin(theta_total + phi)
            D_z = R * math.cos(theta_total + phi)
            x_adj = x_target - D_x
            z_adj = z_target - D_z
            L1 = 0.425
            L2 = 0.39225
            d = math.hypot(x_adj, z_adj)
            if d > L1 + L2 or d < abs(L1 - L2):
                continue
            cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            if cos_theta2 < -1 or cos_theta2 > 1:
                continue
            theta2 = math.acos(cos_theta2)
            cross_product = x_adj * (L1 + L2 * math.cos(theta2)) - z_adj * (L2 * math.sin(theta2))
            if cross_product < 0:
                theta2 = -theta2
            alpha_ik = math.atan2(x_adj, z_adj)
            beta_ik = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
            theta1 = alpha_ik - beta_ik
            theta3 = theta_total - theta1 - theta2
            theta5 = 0.0
            return (theta1, theta2, theta3, theta4, theta5)
    theta1 = math.atan2(x_target, z_target)
    theta2 = 0.0
    theta3 = 0.0
    theta4 = 0.0
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)