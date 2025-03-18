import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes the joint angles needed to reach the given position p for a 6-DOF serial manipulator.
    
    :param p: The desired position as a tuple (x, y, z).
    :return: A tuple of six floats representing the joint angles theta1 to theta6.
    """
    x_d, y_d, z_d = p
    theta1 = math.atan2(x_d, y_d)
    x_adj = x_d - 0.13585 * math.sin(theta1)
    z_adj = z_d - 0.13585 * math.cos(theta1)
    cos_theta2 = z_adj / 0.425
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    x_adj2 = x_adj - 0.1197 * math.sin(theta2)
    z_adj2 = z_adj - 0.1197 * math.cos(theta2)
    theta3 = math.atan2(x_adj2, z_adj2)
    x_adj3 = x_adj2 - 0.39225 * math.sin(theta3)
    z_adj3 = z_adj2 - 0.39225 * math.cos(theta3)
    theta4 = math.atan2(y_d, x_adj3)
    x_adj4 = x_adj3 - 0.093 * math.sin(theta4)
    z_adj4 = z_adj3 - 0.093 * math.cos(theta4)
    theta5 = math.atan2(y_d, x_adj4)
    x_adj5 = x_adj4 - 0.09465 * math.sin(theta5)
    z_adj5 = z_adj4 - 0.09465 * math.cos(theta5)
    theta6 = math.atan2(y_d, x_adj5)
    return (theta1, theta2, theta3, theta4, theta5, theta6)