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
    roll, pitch, yaw = r
    d1 = 0.39225
    d2 = 0.093
    d3 = 0.09465
    tcp_offset = 0.0823
    theta1 = math.atan2(px, pz)
    adjusted_pz = pz - d1 * math.cos(theta1)
    adjusted_px = px - d1 * math.sin(theta1)
    theta2 = math.atan2(adjusted_px, adjusted_pz)
    adjusted_py = py - d2
    theta3 = math.atan2(adjusted_py, adjusted_px)
    adjusted_pz = adjusted_pz - d3
    theta4 = math.atan2(adjusted_px, adjusted_pz)
    return (theta1, theta2, theta3, theta4)