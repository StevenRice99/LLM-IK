import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    y_component = (y - 0.093) / 0.0823
    theta2 = math.acos(y_component)
    for theta2_candidate in [theta2, -theta2]:
        C = -0.0823 * math.sin(theta2_candidate)
        D = 0.09465
        denominator = C ** 2 + D ** 2
        cos_theta1 = (C * x + D * z) / denominator
        sin_theta1 = (D * x - C * z) / denominator
        theta1 = math.atan2(sin_theta1, cos_theta1)
        return (theta1, theta2_candidate, 0.0)
    return (0.0, 0.0, 0.0)