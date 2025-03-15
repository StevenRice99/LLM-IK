import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.093
    theta1 = math.atan2(x_target, z_target)
    x_adjusted = x_target
    z_adjusted = z_target - d1
    r2 = math.sqrt(x_adjusted ** 2 + z_adjusted ** 2)
    theta2 = math.atan2(z_adjusted, x_adjusted) - math.acos(d2 / r2)
    z_adjusted_3 = z_target - (d1 + d2)
    theta3 = math.atan2(z_adjusted_3, x_adjusted) - theta2
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R = Rz @ Ry @ Rx
    theta_pitch = math.atan2(R[2, 0], R[2, 2])
    theta3 += theta_pitch
    return (theta1, theta2, theta3)