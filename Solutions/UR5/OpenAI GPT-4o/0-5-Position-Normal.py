import numpy as np
from sympy import symbols, solve, sin, cos, atan2, sqrt, pi

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    a2 = 0.425
    a3 = 0.39225
    d4 = 0.093
    d6 = 0.09465
    tcp_offset = 0.0823
    z_eff = z - d6
    y_eff = sqrt(y ** 2 + x ** 2) - tcp_offset
    theta1 = atan2(y, x)
    r = sqrt(y_eff ** 2 + z_eff ** 2)
    phi = atan2(z_eff, y_eff)
    cos_theta3 = (r ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    theta3 = atan2(sqrt(1 - cos_theta3 ** 2), cos_theta3)
    theta2 = phi - atan2(a3 * sin(theta3), a2 + a3 * cos(theta3))
    theta4 = 0
    theta5 = 0
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)