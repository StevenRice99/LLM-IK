import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    theta1 = math.atan2(x_target, z_target)
    joint2_x = 0.425 * math.sin(theta1)
    joint2_z = 0.425 * math.cos(theta1)
    A = x_target - joint2_x
    B = z_target - joint2_z
    L1 = 0.39225
    L2 = 0.09465
    numerator = A ** 2 + B ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_theta3 = numerator / denominator
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    A_term = L1 + L2 * math.cos(theta3)
    B_term = L2 * math.sin(theta3)
    denominator_theta2 = A_term ** 2 + B_term ** 2
    sin_theta2 = (A_term * A - B_term * B) / denominator_theta2
    cos_theta2 = (A_term * B + B_term * A) / denominator_theta2
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)