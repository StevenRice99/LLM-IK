import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    target_roll, target_pitch, target_yaw = r
    theta1 = math.atan2(-x, y)
    R_target = np.array([[math.cos(target_yaw) * math.cos(target_pitch), math.cos(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) - math.sin(target_yaw) * math.cos(target_roll), math.cos(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) + math.sin(target_yaw) * math.sin(target_roll)], [math.sin(target_yaw) * math.cos(target_pitch), math.sin(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) + math.cos(target_yaw) * math.cos(target_roll), math.sin(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) - math.cos(target_yaw) * math.sin(target_roll)], [-math.sin(target_pitch), math.cos(target_pitch) * math.sin(target_roll), math.cos(target_pitch) * math.cos(target_roll)]])
    R_theta1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R_theta2 = np.linalg.inv(R_theta1) @ R_target
    theta2 = math.atan2(R_theta2[2, 1], R_theta2[2, 2])
    return (theta1, theta2)