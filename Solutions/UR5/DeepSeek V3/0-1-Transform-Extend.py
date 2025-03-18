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
    phi, theta, psi = r
    theta1 = math.atan2(y, x)
    p_local = np.array([x * math.cos(theta1) + y * math.sin(theta1), -x * math.sin(theta1) + y * math.cos(theta1), z])
    x_local, z_local = (p_local[0], p_local[2])
    theta2 = math.atan2(x_local, z_local) - math.atan2(0.1197, 0.425)
    return (theta1, theta2)