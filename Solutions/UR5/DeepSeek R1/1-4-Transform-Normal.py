import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    r_x, r_y, r_z = r
    x_target, y_target, z_target = p
    cos_rx = math.cos(r_x)
    cos_ry = math.cos(r_y)
    cos_theta_sum = cos_ry * cos_rx
    theta_sum_pos = math.acos(max(min(cos_theta_sum, 1.0), -1.0))
    theta_sum_neg = -theta_sum_pos
    possible_theta_sums = [theta_sum_pos, theta_sum_neg]
    solutions = []
    for theta_sum in possible_theta_sums:
        sin_theta_sum = math.sin(theta_sum)
        cos_theta_sum_val = math.cos(theta_sum)
        numerator_theta4 = math.sin(r_z) * math.cos(r_y)
        denominator_theta4 = math.cos(r_z) / math.cos(r_x) if abs(cos_rx) > 1e-09 else 0.0
        if abs(denominator_theta4) < 1e-09 and abs(numerator_theta4) < 1e-09:
            theta4 = 0.0
        else:
            theta4 = math.atan2(numerator_theta4, denominator_theta4)
        C = 0.09465 * sin_theta_sum
        D_val = 0.09465 * cos_theta_sum_val
        A = 0.425
        B = 0.39225
        numerator = (x_target - C) ** 2 + (z_target - D_val) ** 2 - A ** 2 - B ** 2
        denominator = 2 * A * B
        K = numerator / denominator
        K = max(min(K, 1.0), -1.0)
        try:
            t2 = math.acos(K)
        except ValueError:
            continue
        for current_t2 in [t2, -t2]:
            M = A + B * math.cos(current_t2)
            N = B * math.sin(current_t2)
            denom = M ** 2 + N ** 2
            if denom < 1e-09:
                continue
            sin_t1 = (M * (x_target - C) - N * (z_target - D_val)) / denom
            cos_t1 = (M * (z_target - D_val) + N * (x_target - C)) / denom
            t1 = math.atan2(sin_t1, cos_t1)
            t3 = theta_sum - t1 - current_t2
            solutions.append((t1, current_t2, t3, theta4))
    if not solutions:
        raise ValueError('No valid solution found')
    best_solution = min(solutions, key=lambda sol: min(abs(sol[3] - r_z), abs(sol[3] - r_z + 2 * math.pi), abs(sol[3] - r_z - 2 * math.pi)))
    return best_solution