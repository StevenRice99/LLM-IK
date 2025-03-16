import math
import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta2 = math.acos(z / 0.425)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    K = 0.425 * sin_theta2
    L = 0.01615
    denominator = K ** 2 + L ** 2
    cos_theta1 = (K * x + L * y) / denominator
    sin_theta1 = (-L * x + K * y) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    roll, pitch, yaw = r
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    theta1 += yaw
    theta2 += pitch
    return (theta1, theta2)