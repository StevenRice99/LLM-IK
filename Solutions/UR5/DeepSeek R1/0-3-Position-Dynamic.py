import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    px, py, pz = p
    theta1 = math.atan2(px, -py)
    link2_y_offset = 0.13585
    x_r2 = link2_y_offset * math.sin(theta1)
    y_r2 = link2_y_offset * math.cos(theta1)
    dx = px - x_r2
    dy = py - y_r2
    dz = pz
    total_y_offset = -0.1197 + 0.093
    if not math.isclose(dy, total_y_offset, abs_tol=0.0001):
        dy = total_y_offset
    r = math.hypot(dx, dz)
    planar_z = dy
    theta3, theta4, _ = solve_planar_arm(r, planar_z)
    theta2 = math.atan2(planar_z, r) - (theta3 + theta4)
    return (theta1, theta2, theta3, theta4)

def solve_planar_arm(x_target: float, z_target: float) -> tuple[float, float, float]:
    """Adapted existing code 2 solution for 2-link planar arm"""
    A = 0.425
    B = 0.39225
    numerator = x_target ** 2 + z_target ** 2 - A ** 2 - B ** 2
    denominator = 2 * A * B
    if abs(denominator) < 1e-06:
        return (0.0, 0.0, 0.0)
    cos_theta2 = numerator / denominator
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -theta2_1
    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        C = A + B * math.cos(theta2)
        D = B * math.sin(theta2)
        denominator_theta1 = x_target ** 2 + z_target ** 2
        if abs(denominator_theta1) < 1e-06:
            continue
        sin_theta1 = (C * x_target - D * z_target) / denominator_theta1
        cos_theta1 = (D * x_target + C * z_target) / denominator_theta1
        theta1 = math.atan2(sin_theta1, cos_theta1)
        x_calc = A * math.sin(theta1) + B * math.sin(theta1 + theta2)
        z_calc = A * math.cos(theta1) + B * math.cos(theta1 + theta2)
        error = math.hypot(x_calc - x_target, z_calc - z_target)
        solutions.append((error, theta1, theta2))
    if not solutions:
        return (0.0, 0.0, 0.0)
    solutions.sort()
    best_theta1, best_theta2 = (solutions[0][1], solutions[0][2])
    return (best_theta1, best_theta2, 0.0)