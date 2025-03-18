import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles needed to reach the given position and orientation.
    
    :param p: The desired position as a tuple (x, y, z).
    :param r: The desired orientation in radians as a tuple (roll, pitch, yaw).
    :return: A tuple of two floats representing the joint angles theta1 and theta2.
    """
    x_d, y_d, z_d = p
    r_x, r_y, r_z = r
    theta1 = r_z
    theta2 = r_y
    x = 0.425 * math.cos(theta1) * math.sin(theta2) + 0.1197 * math.sin(theta1)
    y = 0.425 * math.sin(theta1) * math.sin(theta2) - 0.1197 * math.cos(theta1) + 0.13585
    z = 0.425 * math.cos(theta2)
    return (theta1, theta2)