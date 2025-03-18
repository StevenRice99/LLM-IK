import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    :param p: The desired position as a tuple (x, y, z).
    :return: A tuple of three floats representing the joint angles theta1, theta2, and theta3.
    """
    x_d, y_d, z_d = p
    cos_theta2 = (z_d - 0.39225) / 0.425
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    A = 0.425 * math.sin(theta2)
    B = 0.01615
    denominator = A ** 2 + B ** 2
    if denominator == 0:
        denominator = 1e-09
    C = (A * x_d + B * y_d) / denominator
    S = (-B * x_d + A * y_d) / denominator
    theta1 = math.atan2(S, C)
    theta3 = 0.0
    return (theta1, theta2, theta3)