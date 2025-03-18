import numpy as np
from sympy import symbols, Matrix, cos, sin, atan2, sqrt, pi

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.093
    d4 = 0.09465
    tcp_offset = 0.0823
    y -= tcp_offset
    theta1 = atan2(x, z)
    r = sqrt(x ** 2 + z ** 2)
    s = y - d1
    d = sqrt(r ** 2 + s ** 2)
    cos_theta3 = (d2 ** 2 + d3 ** 2 - d ** 2) / (2 * d2 * d3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = atan2(sqrt(1 - cos_theta3 ** 2), cos_theta3)
    cos_theta2 = (d2 ** 2 + d ** 2 - d3 ** 2) / (2 * d2 * d)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = atan2(s, r) - atan2(sqrt(1 - cos_theta2 ** 2), cos_theta2)
    theta4 = 0
    theta5 = 0
    return (float(theta1), float(theta2), float(theta3), float(theta4), float(theta5))