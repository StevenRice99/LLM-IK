import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z]. (Assumed consistent with r)
    :param r: The orientation to reach in radians in the form [x, y, z] (ZYX Euler: roll, pitch, yaw).
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    roll_target, pitch_target, yaw_target = r
    c_roll, s_roll = (math.cos(roll_target), math.sin(roll_target))
    c_pitch, s_pitch = (math.cos(pitch_target), math.sin(pitch_target))
    c_yaw, s_yaw = (math.cos(yaw_target), math.sin(yaw_target))
    R_target_00 = c_yaw * c_pitch
    R_target_val_01 = c_yaw * s_pitch * s_roll - s_yaw * c_roll
    R_target_val_11 = s_yaw * s_pitch * s_roll + c_yaw * c_roll
    R_target_val_20 = -s_pitch
    R_target_val_22 = c_pitch * c_roll
    theta2 = math.atan2(-R_target_val_20, R_target_val_22)
    theta1 = math.atan2(-R_target_val_01, R_target_val_11)
    return (theta1, theta2)