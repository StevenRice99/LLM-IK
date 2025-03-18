import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    :param p: The desired position as a tuple (x, y, z).
    :return: A tuple of six floats representing the joint angles theta1 to theta6.
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
    theta1_rad = theta1
    adjusted_x = x_d - 0.39225 * math.sin(theta1_rad)
    adjusted_z = z_d - 0.39225 * math.cos(theta1_rad)
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta3 = theta_total - theta1_rad
    theta4 = 0.0
    theta5 = 0.0
    theta6 = 0.0
    return (theta1, theta2, theta3, theta4, theta5, theta6)