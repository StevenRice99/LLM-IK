import math
from sympy import symbols, Eq, solve

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    rx, ry, rz = r
    theta1 = math.atan2(p[0], p[2])
    theta2 = math.atan2(p[0], p[2])
    theta3 = math.atan2(p[0], p[2])
    theta4 = math.atan2(p[1], p[0])
    theta5 = math.atan2(p[0], p[2])
    return (theta1, theta2, theta3, theta4, theta5)