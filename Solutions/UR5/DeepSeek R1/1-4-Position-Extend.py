import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    C = 0.09465
    phi = math.atan2(x, z)
    x_adj = x - C * math.sin(phi)
    z_adj = z - C * math.cos(phi)
    d = math.hypot(x_adj, z_adj)
    numerator = d ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    if denominator == 0:
        cos_theta2 = 1.0
    else:
        cos_theta2 = numerator / denominator
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    cross_product = x_adj * (L1 + L2 * math.cos(theta2)) - z_adj * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(x_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    theta_sum = phi
    theta3 = theta_sum - theta1 - theta2
    theta3 = (theta3 + math.pi) % (2 * math.pi) - math.pi
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)