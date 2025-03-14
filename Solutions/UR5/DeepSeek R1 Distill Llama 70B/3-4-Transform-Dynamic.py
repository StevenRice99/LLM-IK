import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.093
    L2 = 0.09465
    theta1 = math.atan2(z - L2, x)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R_desired = np.dot(Rz, np.dot(Ry, Rx))
    Ry_T = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    Rz_theta2 = np.dot(Ry_T, R_desired)
    theta2 = math.atan2(Rz_theta2[1, 0], Rz_theta2[1, 1])
    return (theta1, theta2)