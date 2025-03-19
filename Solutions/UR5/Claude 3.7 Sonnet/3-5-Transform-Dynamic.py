import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    target_roll, target_pitch, target_yaw = r
    theta1 = math.atan2(px, pz)
    R_target = np.array([[math.cos(target_yaw) * math.cos(target_pitch), math.cos(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) - math.sin(target_yaw) * math.cos(target_roll), math.cos(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) + math.sin(target_yaw) * math.sin(target_roll)], [math.sin(target_yaw) * math.cos(target_pitch), math.sin(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) + math.cos(target_yaw) * math.cos(target_roll), math.sin(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) - math.cos(target_yaw) * math.sin(target_roll)], [-math.sin(target_pitch), math.cos(target_pitch) * math.sin(target_roll), math.cos(target_pitch) * math.cos(target_roll)]])
    R_y = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    R_tcp_offset = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_adjusted = R_target @ np.linalg.inv(R_tcp_offset)
    R_in_joint1 = R_y.T @ R_adjusted
    theta2 = math.atan2(R_in_joint1[0, 1], R_in_joint1[0, 0])
    R_z = np.array([[math.cos(theta2), -math.sin(theta2), 0], [math.sin(theta2), math.cos(theta2), 0], [0, 0, 1]])
    R_in_joint2 = R_z.T @ R_in_joint1
    theta3 = math.atan2(-R_in_joint2[0, 2], R_in_joint2[2, 2])
    return (theta1, theta2, theta3)