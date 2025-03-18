import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    distance = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (L1 ** 2 + L2 ** 2 - distance ** 2) / (2 * L1 * L2)
    theta2 = math.acos(cos_theta2)
    A = L1 * math.cos(theta2) + L2
    B = L1 * math.sin(theta2)
    denominator = A ** 2 + B ** 2
    sin_theta = (A * x - B * z) / denominator
    cos_theta = (B * x + A * z) / denominator
    theta = math.atan2(sin_theta, cos_theta)
    theta1 = theta - theta2
    return (theta1, theta2)