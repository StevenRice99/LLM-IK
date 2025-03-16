import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    a = 0.39225
    b = 0.09465
    d_sq = x ** 2 + z ** 2
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    gamma = math.atan2(x, z)
    delta = math.atan2(b * math.sin(theta2), a + b * math.cos(theta2))
    theta1 = gamma - delta
    theta3 = 0.0
    return (theta1, theta2, theta3)