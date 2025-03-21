import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    tcp_offset = 0.09465
    D = math.hypot(x_target, z_target)
    if D == 0:
        return (0.0, 0.0, 0.0, 0.0)
    k = 1 - tcp_offset / D
    x_adj = x_target * k
    z_adj = z_target * k
    d = math.hypot(x_adj, z_adj)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    cross = x_adj * (L1 + L2 * math.cos(theta2)) - z_adj * (L2 * math.sin(theta2))
    if cross < 0:
        theta2 = -theta2
    alpha = math.atan2(x_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    target_angle = math.atan2(x_target, z_target)
    theta3 = target_angle - (theta1 + theta2)
    theta3 = (theta3 + math.pi) % (2 * math.pi) - math.pi
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)