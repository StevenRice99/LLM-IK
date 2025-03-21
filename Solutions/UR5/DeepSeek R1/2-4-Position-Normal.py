import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.39225
    L2 = 0.09465
    numerator = x_target ** 2 + z_target ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_theta2 = numerator / denominator
    theta2 = math.acos(cos_theta2)
    A = L1 + L2 * math.cos(theta2)
    B = L2 * math.sin(theta2)
    denominator_theta1 = A ** 2 + B ** 2
    sin_theta1 = (A * x_target - B * z_target) / denominator_theta1
    cos_theta1 = (B * x_target + A * z_target) / denominator_theta1
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta3 = 0.0
    return (theta1, theta2, theta3)