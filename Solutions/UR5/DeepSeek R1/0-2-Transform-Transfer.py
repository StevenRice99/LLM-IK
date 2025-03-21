import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = yaw
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    x_prime = x * cos_theta1 + y * sin_theta1
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    A = 0.425 - 0.39225 * cos_p
    B = 0.39225 * sin_p
    C = 0.425 + 0.39225 * cos_p
    D = B
    det = A * C - B * D
    if abs(det) < 1e-06:
        raise ValueError('Singular matrix, target not reachable')
    sin_theta2 = (C * x_prime - B * z) / det
    cos_theta2 = (A * z - D * x_prime) / det
    norm = math.hypot(sin_theta2, cos_theta2)
    sin_theta2 /= norm
    cos_theta2 /= norm
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta3 = pitch - theta2
    return (theta1, theta2, theta3)