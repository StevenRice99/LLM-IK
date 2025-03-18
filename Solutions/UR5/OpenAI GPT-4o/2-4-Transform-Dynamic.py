import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    d1 = 0.39225
    d2 = 0.093
    d3 = 0.09465
    theta1 = math.atan2(x, z)
    x_prime = x - d2 * math.sin(theta1)
    z_prime = z - d2 * math.cos(theta1)
    theta2 = math.atan2(x_prime, z_prime)
    y_prime = y - d3
    theta3 = math.atan2(y_prime, x_prime)
    return (theta1, theta2, theta3)