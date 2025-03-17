import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    dir_to_target = math.atan2(x_target, z_target)
    x_adj = x_target - L3 * math.sin(dir_to_target)
    z_adj = z_target - L3 * math.cos(dir_to_target)
    r_adj = math.hypot(x_adj, z_adj)
    cos_theta2 = (r_adj ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -theta2_1
    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        beta = math.atan2(x_adj, z_adj)
        alpha = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
        theta1 = (beta - alpha) % (2 * math.pi)
        x2 = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
        z2 = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        dx = x_target - x2
        dz = z_target - z2
        theta_residual = math.atan2(dx, dz)
        theta3 = (theta_residual - (theta1 + theta2)) % (2 * math.pi)
        if theta3 > math.pi:
            theta3 -= 2 * math.pi
        solutions.append((theta1, theta2, theta3))
    theta1, theta2, theta3 = min(solutions, key=lambda s: abs(s[2]))
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    theta3 = (theta3 + math.pi) % (2 * math.pi) - math.pi
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)