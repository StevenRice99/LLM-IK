import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x_d, y_d, z_d = p
    r_x, r_y, r_z = r
    cos_theta2 = z_d / 0.425
    sin_theta2 = math.sin(r_y)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    A = 0.425 * math.sin(theta2)
    B = 0.01615
    denominator = A ** 2 + B ** 2
    C = (A * x_d + B * y_d) / denominator
    S = (-B * x_d + A * y_d) / denominator
    theta1 = math.atan2(S, C)
    return (theta1, theta2)