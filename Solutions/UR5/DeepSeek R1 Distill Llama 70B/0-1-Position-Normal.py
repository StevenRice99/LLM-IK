import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    x_des, y_des, z_des = p
    z_scale = z_des / 0.425
    if abs(z_scale) > 1.0:
        return (0.0, 0.0)
    theta2_candidates = [math.acos(z_scale), -math.acos(z_scale)]
    solutions = []
    for theta2 in theta2_candidates:
        theta2_adjusted = theta2
        if theta2_adjusted > math.pi:
            theta2_adjusted -= 2 * math.pi
        elif theta2_adjusted < -math.pi:
            theta2_adjusted += 2 * math.pi
        sin_theta2 = math.sin(theta2_adjusted)
        cos_theta2 = math.cos(theta2_adjusted)
        cos_theta1 = (x_des + 0.425 * sin_theta2) / 0.13585
        sin_theta1 = (y_des + 0.1197) / 0.13585
        if abs(cos_theta1) > 1.0 or abs(sin_theta1) > 1.0:
            continue
        theta1 = math.atan2(sin_theta1, cos_theta1)
        x_fk = 0.13585 * math.cos(theta1) - 0.425 * math.sin(theta2_adjusted)
        y_fk = 0.13585 * math.sin(theta1) - 0.1197
        z_fk = 0.425 * math.cos(theta2_adjusted)
        if math.isclose(x_fk, x_des, rel_tol=1e-05) and math.isclose(y_fk, y_des, rel_tol=1e-05) and math.isclose(z_fk, z_des, rel_tol=1e-05):
            solutions.append((theta1, theta2_adjusted))
    if solutions:
        return (solutions[0][0], solutions[0][1])
    else:
        return (0.0, 0.0)