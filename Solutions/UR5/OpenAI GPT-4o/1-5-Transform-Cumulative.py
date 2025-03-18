import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    roll, pitch, yaw = r
    theta1 = math.atan2(-px, pz)
    d1 = 0.425
    adjusted_py = py + 0.1197
    adjusted_pz = pz - d1
    theta2 = math.atan2(adjusted_py, adjusted_pz)
    d2 = 0.39225
    adjusted_pz_3 = adjusted_pz - d2
    theta3 = math.atan2(adjusted_py, adjusted_pz_3)
    theta4 = math.atan2(py, px)
    theta5 = yaw - theta4
    return (theta1, theta2, theta3, theta4, theta5)