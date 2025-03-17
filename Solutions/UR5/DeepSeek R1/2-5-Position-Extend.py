import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    a = 0.39225
    b = 0.09465
    y_offset_revolute3 = 0.093
    y_tcp = 0.0823
    total_y_offset = y_offset_revolute3 + y_tcp
    theta3 = math.atan2(y_target - y_offset_revolute3, x_target)
    cos_theta3 = math.cos(theta3)
    sin_theta3 = math.sin(theta3)
    x_rot = x_target * cos_theta3 + (y_target - y_offset_revolute3) * sin_theta3
    z_eff = z_target - b
    d_sq = x_rot ** 2 + z_eff ** 2
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    solutions = []
    for theta2_sign in [1, -1]:
        current_theta2 = theta2 * theta2_sign
        gamma = math.atan2(x_rot, z_eff)
        delta = math.atan2(b * math.sin(current_theta2), a + b * math.cos(current_theta2))
        theta1 = gamma - delta
        solutions.append((theta1, current_theta2, theta3))
    if not solutions:
        raise ValueError('No valid solution found')
    best_solution = min(solutions, key=lambda sol: abs(sol[1]))
    theta1, theta2, theta3 = best_solution
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)