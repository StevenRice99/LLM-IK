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
    d1 = 0.1197
    a2 = 0.425
    a3 = 0.39225
    d4 = 0.093
    d_tcp = 0.09465
    theta1 = math.atan2(px, pz)
    r = math.sqrt(px ** 2 + pz ** 2)
    z = pz - d1
    d = math.sqrt(r ** 2 + z ** 2)
    if d > a2 + a3:
        raise ValueError('Target is out of reach')
    cos_theta2 = (a2 ** 2 + d ** 2 - a3 ** 2) / (2 * a2 * d)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(z, r) - math.atan2(a3 * sin_theta2, a2 + a3 * cos_theta2)
    cos_theta3 = (a2 ** 2 + a3 ** 2 - d ** 2) / (2 * a2 * a3)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    theta4 = rz
    return (theta1, theta2, theta3, theta4)