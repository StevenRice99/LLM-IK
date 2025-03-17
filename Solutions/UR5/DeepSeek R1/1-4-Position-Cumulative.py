import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    D = math.hypot(x_target, z_target)
    if D == 0:
        return (0.0, 0.0, 0.0, 0.0)
    R_min = abs(L1 - L2)
    R_max = L1 + L2
    solutions = []
    for r in [R_min, R_max]:
        a = (r ** 2 - L3 ** 2 + D ** 2) / (2 * D)
        h_sq = r ** 2 - a ** 2
        if h_sq < 0:
            continue
        h = math.sqrt(h_sq)
        x_eff1 = (a * x_target + h * z_target) / D
        z_eff1 = (a * z_target - h * x_target) / D
        x_eff2 = (a * x_target - h * z_target) / D
        z_eff2 = (a * z_target + h * x_target) / D
        for x_eff, z_eff in [(x_eff1, z_eff1), (x_eff2, z_eff2)]:
            r_eff = math.hypot(x_eff, z_eff)
            if r_eff < R_min or r_eff > R_max:
                continue
            cos_theta2 = (r_eff ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
            theta2 = math.acos(cos_theta2)
            for theta2_val in [theta2, -theta2]:
                beta = math.atan2(x_eff, z_eff)
                alpha = math.atan2(L2 * math.sin(theta2_val), L1 + L2 * math.cos(theta2_val))
                theta1 = beta - alpha
                dx = x_target - (L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2_val))
                dz = z_target - (L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2_val))
                gamma = math.atan2(dx, dz)
                theta3 = gamma - (theta1 + theta2_val)
                solutions.append((theta1, theta2_val, theta3))
    if not solutions:
        return (0.0, 0.0, 0.0, 0.0)
    valid_solutions = []
    for sol in solutions:
        theta1, theta2, theta3 = sol
        x3 = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
        z3 = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        x_tcp = x3 + L3 * math.sin(theta1 + theta2 + theta3)
        z_tcp = z3 + L3 * math.cos(theta1 + theta2 + theta3)
        if math.isclose(x_tcp, x_target, abs_tol=0.0001) and math.isclose(z_tcp, z_target, abs_tol=0.0001):
            valid_solutions.append(sol)
    if valid_solutions:
        theta1, theta2, theta3 = min(valid_solutions, key=lambda s: abs(s[1]))
    else:
        theta1, theta2, theta3 = solutions[0]
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)