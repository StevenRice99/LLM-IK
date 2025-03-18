import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    phi, theta, psi = r
    A = 0.13585
    B = 0.1197
    C = 0.425
    alpha = math.atan2(B, A)
    denominator = math.sqrt(A ** 2 + B ** 2)
    sin_theta2_alpha = max(-1, min(1, z / denominator))
    theta2 = math.asin(sin_theta2_alpha) - alpha
    D = A * math.cos(theta2) - B * math.sin(theta2)
    numerator = y * D + C * x
    denominator = x * D - C * y
    theta1 = math.atan2(numerator, denominator)
    return (theta1, theta2)