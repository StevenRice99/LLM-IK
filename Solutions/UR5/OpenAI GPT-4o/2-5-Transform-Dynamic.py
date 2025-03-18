import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    px = math.sqrt(x ** 2 + z ** 2)
    py = y
    theta2 = math.atan2(px, py)
    target_roll, target_pitch, target_yaw = r
    theta3, theta4 = inverse_kinematics_joints_3_and_4(p, r)
    return (theta1, theta2, theta3, theta4)

def inverse_kinematics_joints_3_and_4(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Helper function to solve for Joints 3 and 4.
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
    theta3 = math.atan2(R_theta2[2, 1], R_theta2[2, 2])
    theta4 = math.atan2(R_theta2[1, 0], R_theta2[0, 0])
    return (theta3, theta4)