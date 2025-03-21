import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    cx = math.cos(rx)
    sx = math.sin(rx)
    cy = math.cos(ry)
    sy = math.sin(ry)
    cz = math.cos(rz)
    sz = math.sin(rz)
    R00 = cy * cz
    R02 = sy
    theta_sum = math.atan2(R02, R00)
    a = 0.425
    b = 0.39225
    d_sq = x_target ** 2 + z_target ** 2
    d = math.sqrt(d_sq)
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -theta2_1
    solutions = []
    for theta2 in [theta2_1, theta2_2]:
        theta1 = theta_sum - theta2
        x_calc = a * math.sin(theta1) + b * math.sin(theta_sum)
        z_calc = a * math.cos(theta1) + b * math.cos(theta_sum)
        if math.isclose(x_calc, x_target, abs_tol=1e-05) and math.isclose(z_calc, z_target, abs_tol=1e-05):
            solutions.append((theta1, theta2))
    if solutions:
        for theta1, theta2 in solutions:
            if -6.2831853 <= theta2 <= 6.2831853:
                return (theta1, theta2)
        return solutions[0]
    else:
        term_x = x_target - b * math.sin(theta_sum)
        term_z = z_target - b * math.cos(theta_sum)
        theta1 = math.atan2(term_x, term_z)
        theta2 = theta_sum - theta1
        return (theta1, theta2)