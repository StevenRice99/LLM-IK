import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    a = 0.0823
    L1 = 0.093
    L2 = 0.09465
    c2 = (y_target - L1) / a
    if abs(c2) > 1:
        raise ValueError('Target position is unreachable')
    s2 = math.sqrt(1 - c2 ** 2)
    theta2_options = [math.acos(c2), -math.acos(c2)]
    best_solution = None
    min_error = float('inf')
    for theta2 in theta2_options:
        current_s2 = math.sin(theta2)
        A = L2
        B = -a * current_s2
        denominator = A ** 2 + (a * current_s2) ** 2
        if denominator < 1e-10:
            continue
        sin_theta1 = (A * x_target - B * z_target) / denominator
        cos_theta1 = (A * z_target + B * x_target) / denominator
        if abs(sin_theta1 ** 2 + cos_theta1 ** 2 - 1) > 1e-06:
            continue
        theta1 = math.atan2(sin_theta1, cos_theta1)
        x = L2 * sin_theta1 - a * current_s2 * cos_theta1
        z = L2 * cos_theta1 + a * current_s2 * sin_theta1
        error = (x - x_target) ** 2 + (z - z_target) ** 2
        if error < min_error:
            min_error = error
            best_solution = (theta1, theta2, 0.0)
    if best_solution is None:
        raise ValueError('Target position is unreachable')
    return best_solution