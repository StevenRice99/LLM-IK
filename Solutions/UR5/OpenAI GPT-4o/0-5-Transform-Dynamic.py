import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = math.atan2(-x, y)
    d2 = 0.13585
    z2 = z - 0
    theta2 = math.atan2(x, z2)
    d3 = 0.425
    z3 = z2 - d2
    theta3 = math.atan2(x, z3)
    d4 = 0.39225
    z4 = z3 - d3
    theta4 = math.atan2(x, z4)
    R_target = np.array([[math.cos(yaw) * math.cos(pitch), math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll), math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)], [math.sin(yaw) * math.cos(pitch), math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll), math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)], [-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)]])
    R_theta1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    R_theta2 = np.linalg.inv(R_theta1) @ R_target
    theta5 = math.atan2(R_theta2[2, 1], R_theta2[2, 2])
    theta6 = math.atan2(-R_theta2[2, 0], math.sqrt(R_theta2[2, 1] ** 2 + R_theta2[2, 2] ** 2))
    return (theta1, theta2, theta3, theta4, theta5, theta6)