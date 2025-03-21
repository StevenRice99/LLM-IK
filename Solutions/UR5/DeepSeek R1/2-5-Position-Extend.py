import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.39225
    L2 = 0.09465
    L3 = 0.093
    L4 = 0.0823
    y_contribution = y_target - L3
    cos_theta3 = y_contribution / L4
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    delta_x = L4 * math.sin(theta3)
    delta_z = L2
    adjusted_x = x_target - delta_x
    adjusted_z = z_target - delta_z
    numerator = adjusted_x ** 2 + adjusted_z ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_theta2 = numerator / denominator
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    A = L1 + L2 * math.cos(theta2)
    B = L2 * math.sin(theta2)
    denominator_theta1 = A ** 2 + B ** 2
    sin_theta1 = (A * adjusted_x - B * adjusted_z) / denominator_theta1
    cos_theta1 = (B * adjusted_x + A * adjusted_z) / denominator_theta1
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)