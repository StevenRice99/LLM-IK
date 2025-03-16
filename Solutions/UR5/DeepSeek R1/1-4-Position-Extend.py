import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    A = 0.425
    B = 0.39225
    L3 = 0.09465
    theta_sum = math.atan2(x_target, z_target)
    x_adj = x_target - L3 * math.sin(theta_sum)
    z_adj = z_target - L3 * math.cos(theta_sum)
    r_adj = math.hypot(x_adj, z_adj)
    cos_theta2 = (r_adj ** 2 - A ** 2 - B ** 2) / (2 * A * B)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -theta2_1
    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        beta = math.atan2(x_adj, z_adj)
        alpha = math.atan2(B * math.sin(theta2), A + B * math.cos(theta2))
        theta1 = (beta - alpha) % (2 * math.pi)
        theta3 = (theta_sum - (theta1 + theta2)) % (2 * math.pi)
        if theta3 > math.pi:
            theta3 -= 2 * math.pi
        solutions.append((theta1, theta2, theta3))
    theta1, theta2, theta3 = min(solutions, key=lambda s: abs(s[2]))
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)