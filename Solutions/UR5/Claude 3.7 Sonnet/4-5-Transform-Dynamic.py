import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.09465
    L2 = 0.0823
    joint1 = math.atan2(y, x)
    x_local = x * math.cos(-joint1) - y * math.sin(-joint1)
    y_local = x * math.sin(-joint1) + y * math.cos(-joint1)
    z_local = z
    joint2 = math.atan2(x_local - L2 * math.sin(rz - joint1), z_local - L1)
    return (joint1, joint2)