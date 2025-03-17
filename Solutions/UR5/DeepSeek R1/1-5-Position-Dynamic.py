import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    a = 0.425
    theta1 = math.atan2(x_target, z_target) - math.atan2(a, math.hypot(x_target, z_target))
    x_r2 = a * math.sin(theta1)
    y_r2 = -0.1197
    z_r2 = a * math.cos(theta1)
    x_rel = x_target - x_r2
    y_rel = y_target - y_r2
    z_rel = z_target - z_r2
    cos_t1 = math.cos(theta1)
    sin_t1 = math.sin(theta1)
    x_prime = x_rel * cos_t1 + z_rel * sin_t1
    z_prime = -x_rel * sin_t1 + z_rel * cos_t1
    y_prime = y_rel
    A = 0.39225
    B = 0.09465
    C_tcp = 0.0823
    y_offset = y_prime - 0.093
    if abs(y_offset) > C_tcp + 1e-06:
        raise ValueError('Target y is out of reach')
    theta3 = math.acos(max(min(y_offset / C_tcp, 1.0), -1.0))
    sin_t3 = math.sin(theta3)
    C = C_tcp * sin_t3
    M = B * x_prime + C * z_prime
    N = B * z_prime - C * x_prime
    K = (x_prime ** 2 + z_prime ** 2 + B ** 2 + C ** 2 - A ** 2) / 2
    denominator = math.hypot(M, N)
    if denominator < 1e-06:
        raise ValueError('Singular position')
    ratio = K / denominator
    if abs(ratio) > 1.0:
        if abs(ratio) - 1.0 < 1e-06:
            ratio = math.copysign(1.0, ratio)
        else:
            raise ValueError('Target xz is out of reach')
    phi = math.atan2(N, M)
    theta_sum = math.asin(ratio) - phi
    solutions = []
    for theta_sum_candidate in [theta_sum, math.pi - theta_sum - 2 * phi]:
        sin_ts = math.sin(theta_sum_candidate)
        cos_ts = math.cos(theta_sum_candidate)
        D = B * sin_ts - C * cos_ts
        E = B * cos_ts + C * sin_ts
        sin_t1p = (x_prime - D) / A
        cos_t1p = (z_prime - E) / A
        if abs(sin_t1p ** 2 + cos_t1p ** 2 - 1) < 1e-06:
            solutions.append((theta_sum_candidate, math.atan2(sin_t1p, cos_t1p)))
    if not solutions:
        raise ValueError('No valid solution found')
    theta_sum, theta1_prime = solutions[0]
    theta2 = theta_sum - theta1_prime
    return (theta1, theta2, theta3, 0.0, 0.0)