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
    rx, ry, rz = r
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    theta1 = math.atan2(x, z)
    wrist_x = x - l3 * math.sin(ry) * math.cos(rz)
    wrist_y = y - l2 - l3 * math.sin(rz)
    wrist_z = z - l3 * math.cos(ry) * math.cos(rz)
    proj_xz = math.sqrt(wrist_x ** 2 + wrist_z ** 2)
    theta2 = math.atan2(proj_xz, l1 - wrist_z / math.cos(theta1) if theta1 != math.pi / 2 else l1 - wrist_x)
    theta3 = rz
    return (theta1, theta2, theta3)