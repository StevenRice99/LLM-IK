import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    y_offset = y_target - 0.093
    if abs(y_offset) > 0.0823:
        raise ValueError('Target y is out of reach')
    theta3 = math.acos(y_offset / 0.0823)
    sin_theta3 = math.sin(theta3)
    cos_theta3 = math.cos(theta3)
    A = 0.39225
    B = 0.09465
    C = 0.0823 * sin_theta3
    M_val = B * x_target + C * z_target
    N_val = B * z_target - C * x_target
    K_val = (x_target ** 2 + z_target ** 2 + B ** 2 + C ** 2 - A ** 2) / 2
    denominator = math.hypot(M_val, N_val)
    if abs(K_val) > denominator:
        raise ValueError('Target xz is out of reach')
    phi = math.atan2(N_val, M_val)
    theta_sum = math.asin(K_val / denominator) - phi
    theta_sum1 = theta_sum
    theta_sum2 = math.pi - theta_sum - 2 * phi
    sin_theta_sum1 = math.sin(theta_sum1)
    cos_theta_sum1 = math.cos(theta_sum1)
    D1 = B * sin_theta_sum1 - C * cos_theta_sum1
    E1 = B * cos_theta_sum1 + C * sin_theta_sum1
    sin_theta1_1 = (x_target - D1) / A
    cos_theta1_1 = (z_target - E1) / A
    sin_theta_sum2 = math.sin(theta_sum2)
    cos_theta_sum2 = math.cos(theta_sum2)
    D2 = B * sin_theta_sum2 - C * cos_theta_sum2
    E2 = B * cos_theta_sum2 + C * sin_theta_sum2
    sin_theta1_2 = (x_target - D2) / A
    cos_theta1_2 = (z_target - E2) / A
    valid1 = abs(sin_theta1_1 ** 2 + cos_theta1_1 ** 2 - 1) < 1e-06
    valid2 = abs(sin_theta1_2 ** 2 + cos_theta1_2 ** 2 - 1) < 1e-06
    if valid1 and valid2:
        theta1_1 = math.atan2(sin_theta1_1, cos_theta1_1)
        theta1_2 = math.atan2(sin_theta1_2, cos_theta1_2)
        theta1 = theta1_1
        theta_sum = theta_sum1
    elif valid1:
        theta1 = math.atan2(sin_theta1_1, cos_theta1_1)
        theta_sum = theta_sum1
    elif valid2:
        theta1 = math.atan2(sin_theta1_2, cos_theta1_2)
        theta_sum = theta_sum2
    else:
        raise ValueError('No valid solution found')
    theta2 = theta_sum - theta1
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)