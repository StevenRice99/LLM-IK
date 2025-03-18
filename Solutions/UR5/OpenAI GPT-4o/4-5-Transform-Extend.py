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
    cos_theta_2 = (z - L1) / L2
    theta_2 = math.acos(cos_theta_2)
    theta_1 = math.atan2(y, x)
    R_z_theta1 = np.array([[math.cos(theta_1), -math.sin(theta_1), 0], [math.sin(theta_1), math.cos(theta_1), 0], [0, 0, 1]])
    R_y_theta2 = np.array([[math.cos(theta_2), 0, math.sin(theta_2)], [0, 1, 0], [-math.sin(theta_2), 0, math.cos(theta_2)]])
    R_current = np.dot(R_z_theta1, R_y_theta2)
    R_target = np.array([[math.cos(rz) * math.cos(ry), -math.sin(rz) * math.cos(ry), math.sin(ry)], [math.sin(rz) * math.cos(rx) + math.cos(rz) * math.sin(ry) * math.sin(rx), math.cos(rz) * math.cos(rx) - math.sin(rz) * math.sin(ry) * math.sin(rx), -math.cos(ry) * math.sin(rx)], [math.sin(rz) * math.sin(rx) - math.cos(rz) * math.sin(ry) * math.cos(rx), math.cos(rz) * math.sin(rx) + math.sin(rz) * math.sin(ry) * math.cos(rx), math.cos(ry) * math.cos(rx)]])
    R_error = np.dot(R_target, R_current.T)
    theta_1_correction = math.atan2(R_error[1, 0], R_error[0, 0])
    theta_2_correction = math.atan2(R_error[2, 1], R_error[2, 2])
    theta_1 += theta_1_correction
    theta_2 += theta_2_correction
    return (theta_1, theta_2)