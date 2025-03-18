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
    theta_1 = math.atan2(x, z)
    y_effective = y - 0.093
    theta_2 = math.asin(y_effective / 0.09465)
    R_desired = np.array([[math.cos(rz) * math.cos(ry), math.cos(rz) * math.sin(ry) * math.sin(rx) - math.sin(rz) * math.cos(rx), math.cos(rz) * math.sin(ry) * math.cos(rx) + math.sin(rz) * math.sin(rx)], [math.sin(rz) * math.cos(ry), math.sin(rz) * math.sin(ry) * math.sin(rx) + math.cos(rz) * math.cos(rx), math.sin(rz) * math.sin(ry) * math.cos(rx) - math.cos(rz) * math.sin(rx)], [-math.sin(ry), math.cos(ry) * math.sin(rx), math.cos(ry) * math.cos(rx)]])
    R1 = np.array([[math.cos(theta_1), 0, math.sin(theta_1)], [0, 1, 0], [-math.sin(theta_1), 0, math.cos(theta_1)]])
    R2 = np.array([[math.cos(theta_2), -math.sin(theta_2), 0], [math.sin(theta_2), math.cos(theta_2), 0], [0, 0, 1]])
    R_current = R1 @ R2
    theta_1_refined = math.atan2(R_desired[2, 0], R_desired[2, 2])
    theta_2_refined = math.atan2(R_desired[0, 1], R_desired[1, 1])
    return (theta_1_refined, theta_2_refined)