import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1 = math.atan2(y, x)
    adjusted_x = x - 0.0823 * math.cos(theta1)
    adjusted_y = y - 0.0823 * math.sin(theta1)
    adjusted_z = z - 0.09465
    theta2 = math.atan2(adjusted_x, adjusted_z)
    theta2 += ry
    return (theta1, theta2)