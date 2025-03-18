import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    roll, pitch, yaw = r
    theta1 = math.atan2(px, pz)
    R1_inv = np.array([[math.cos(-theta1), 0, math.sin(-theta1)], [0, 1, 0], [-math.sin(-theta1), 0, math.cos(-theta1)]])
    adjusted_p = R1_inv @ np.array([px, py, pz])
    adjusted_px, adjusted_py, adjusted_pz = adjusted_p
    theta2 = math.atan2(adjusted_px, adjusted_py - 0.093)
    R_target = np.array([[math.cos(yaw) * math.cos(pitch), math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll), math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)], [math.sin(yaw) * math.cos(pitch), math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll), math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)], [-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)]])
    R2 = np.array([[math.cos(theta2), -math.sin(theta2), 0], [math.sin(theta2), math.cos(theta2), 0], [0, 0, 1]])
    R3 = np.linalg.inv(R2) @ R_target
    theta3 = math.atan2(R3[2, 1], R3[2, 2])
    return (theta1, theta2, theta3)