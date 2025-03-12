import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    A = 0.0823
    B = 0.09465
    cos_theta2 = z / B
    if abs(cos_theta2) > 1:
        raise ValueError('Target position is not reachable')
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.acos(cos_theta2)
    cos_theta1 = y / A
    if abs(cos_theta1) > 1:
        raise ValueError('Target position is not reachable')
    sin_theta1 = math.sqrt(1 - cos_theta1 ** 2)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)