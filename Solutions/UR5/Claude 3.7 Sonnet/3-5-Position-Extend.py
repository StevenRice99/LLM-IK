import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    l1 = 0.093
    l2 = 0.09465
    l3 = 0.0823
    theta1 = math.atan2(px, pz)
    r_xz = math.sqrt(px ** 2 + pz ** 2)
    cos_theta2 = (py - l1) / l3
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta3 = (r_xz - l3 * math.sin(theta2)) / l2
    sin_theta3 = max(min(sin_theta3, 1.0), -1.0)
    theta3 = math.asin(sin_theta3)
    return (theta1, theta2, theta3)