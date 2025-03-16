import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    theta1 = math.atan2(y, x)
    r = math.hypot(x, y) - 0.13585
    z_rel = z
    L1 = math.hypot(0.1197, 0.425)
    L23 = 0.39225 + 0.093
    d = math.hypot(r, z_rel)
    cos_theta3 = (L1 ** 2 + L23 ** 2 - d ** 2) / (2 * L1 * L23)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    alpha = math.atan2(z_rel, r)
    beta = math.asin(L23 * math.sin(-theta3) / d)
    theta2 = alpha - beta
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)