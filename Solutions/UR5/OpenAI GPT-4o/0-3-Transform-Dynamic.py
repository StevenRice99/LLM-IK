import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    d1 = 0.13585
    d2 = 0.425
    d3 = 0.39225
    d_tcp = 0.093
    theta1 = math.atan2(py, px)
    wx = px - d_tcp * math.cos(theta1)
    wy = py - d_tcp * math.sin(theta1)
    wz = pz
    r_wrist = math.sqrt(wx ** 2 + wy ** 2)
    z_wrist = wz - d1
    a = d2
    b = d3
    c = math.sqrt(r_wrist ** 2 + z_wrist ** 2)
    cos_theta2 = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(z_wrist, r_wrist) - math.atan2(a * sin_theta2, a * cos_theta2)
    cos_theta3 = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    theta4 = rz - (theta2 + theta3)
    return (theta1, theta2, theta3, theta4)