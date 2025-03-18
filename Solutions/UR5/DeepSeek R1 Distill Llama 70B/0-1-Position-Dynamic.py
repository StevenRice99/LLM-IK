import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    :param p: The desired position as a tuple (x, y, z).
    :return: A tuple of two floats representing the joint angles theta1 and theta2.
    """
    x_d, y_d, z_d = p
    cos_theta2 = z_d / 0.425
    theta2 = math.acos(cos_theta2)
    A = 0.425 * math.sin(theta2)
    B = 0.01615
    denominator = A ** 2 + B ** 2
    C = (A * x_d + B * y_d) / denominator
    S = (-B * x_d + A * y_d) / denominator
    theta1 = math.atan2(S, C)
    return (theta1, theta2)