import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    l1 = 0.425
    l2 = 0.39225
    A = 0.09465
    B_tcp = 0.0823
    r = math.hypot(x_target, z_target)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    beta = math.atan2(x_target, z_target)
    alpha = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta - alpha
    x_j4 = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    z_j4 = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y_j4 = -0.1197 + 0.093
    dx = x_target - x_j4
    dy = y_target - y_j4
    dz = z_target - z_j4
    if abs(dz) > B_tcp:
        raise ValueError('Target z is out of reach')
    theta5 = -math.asin(dz / B_tcp)
    cos_theta5 = math.cos(theta5)
    B = B_tcp * cos_theta5
    a = 0.09465
    denominator = a ** 2 - B ** 2
    if abs(denominator) < 1e-09:
        raise ValueError('Singular matrix; target xyz is out of reach')
    sin_theta4 = (a * dx - B * dy) / denominator
    cos_theta4 = (-B * dx + a * dy) / denominator
    norm = math.hypot(sin_theta4, cos_theta4)
    sin_theta4 /= norm
    cos_theta4 /= norm
    theta4 = math.atan2(sin_theta4, cos_theta4)
    theta3 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)