import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Computes the joint angles needed to reach the given position p for a 4-DOF serial manipulator.
    
    :param p: The desired position as a tuple (x, y, z).
    :return: A tuple of four floats representing the joint angles theta1, theta2, theta3, and theta4.
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    adjusted_x = x - 0.13585 * math.sin(theta1)
    adjusted_z = z - 0.13585 * math.cos(theta1)
    a = 0.425
    b = 0.39225
    c = math.sqrt(adjusted_x ** 2 + adjusted_z ** 2)
    acos_arg_theta2 = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    acos_arg_theta2 = max(min(acos_arg_theta2, 1.0), -1.0)
    theta2 = math.acos(acos_arg_theta2)
    acos_arg_theta3 = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    acos_arg_theta3 = max(min(acos_arg_theta3, 1.0), -1.0)
    theta3 = math.acos(acos_arg_theta3)
    theta4 = math.atan2(adjusted_x, adjusted_z) - (theta2 + theta3)
    theta1 = max(min(theta1, math.pi), -math.pi)
    theta2 = max(min(theta2, math.pi / 2), -math.pi / 2)
    theta3 = max(min(theta3, math.pi / 2), -math.pi / 2)
    theta4 = max(min(theta4, math.pi), -math.pi)
    return (theta1, theta2, theta3, theta4)