import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    a = 0.39225
    b = 0.09465
    d = 0.093
    d_sq = x_tcp ** 2 + z_tcp ** 2
    cos_theta4 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta4 = max(min(cos_theta4, 1.0), -1.0)
    theta4 = math.acos(cos_theta4)
    gamma = math.atan2(x_tcp, z_tcp)
    delta = math.atan2(b * math.sin(theta4), a + b * math.cos(theta4))
    theta3 = gamma - delta
    wrist_x = x_tcp - b * math.sin(theta4)
    wrist_y = y_tcp - d
    wrist_z = z_tcp - b * math.cos(theta4)
    a1 = 0.13585
    a2 = 0.425
    a3 = -0.1197
    r = math.hypot(wrist_x, wrist_z)
    theta1 = math.atan2(wrist_x, wrist_z)
    y_proj = wrist_y - a1
    c = math.hypot(y_proj, r)
    if c == 0:
        raise ValueError('Unreachable position')
    if c > a2 + abs(a3) or c < abs(a2 - abs(a3)):
        raise ValueError('Unreachable position')
    cos_theta3 = (a2 ** 2 + c ** 2 - a3 ** 2) / (2 * a2 * c)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3_val = math.acos(cos_theta3)
    theta3_alt = -theta3_val
    solutions = []
    for theta3 in [theta3_val, theta3_alt]:
        alpha = math.atan2(y_proj, r)
        beta = math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3))
        theta2 = alpha - beta
        solutions.append((theta1, theta2, theta3))
    if not solutions:
        raise ValueError('Unreachable position')
    theta1, theta2, theta3 = solutions[0]
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)