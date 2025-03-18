import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    roll, pitch, yaw = r
    theta1 = math.atan2(px, pz)
    d2 = 0.1197
    d3 = 0.39225
    l2 = 0.425
    l3 = 0.39225
    yz_distance = math.sqrt(py ** 2 + (pz - d2) ** 2)
    cos_theta2 = (l2 ** 2 + yz_distance ** 2 - l3 ** 2) / (2 * l2 * yz_distance)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.atan2(py, pz - d2) - math.acos(cos_theta2)
    cos_theta3 = (l2 ** 2 + l3 ** 2 - yz_distance ** 2) / (2 * l2 * l3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    R_target = np.array([[math.cos(yaw) * math.cos(pitch), math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll), math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)], [math.sin(yaw) * math.cos(pitch), math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll), math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)], [-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)]])
    R_theta1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R_theta2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R_effective = np.linalg.inv(R_theta1 @ R_theta2) @ R_target
    theta4 = math.atan2(R_effective[1, 0], R_effective[0, 0])
    theta5 = math.atan2(-R_effective[2, 0], math.sqrt(R_effective[2, 1] ** 2 + R_effective[2, 2] ** 2))
    return (theta1, theta2, theta3, theta4, theta5)