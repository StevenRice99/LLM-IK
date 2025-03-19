def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    target_roll, target_pitch, target_yaw = r
    tcp_offset = 0.0823
    R_target = np.array([[math.cos(target_yaw) * math.cos(target_pitch), math.cos(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) - math.sin(target_yaw) * math.cos(target_roll), math.cos(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) + math.sin(target_yaw) * math.sin(target_roll)], [math.sin(target_yaw) * math.cos(target_pitch), math.sin(target_yaw) * math.sin(target_pitch) * math.sin(target_roll) + math.cos(target_yaw) * math.cos(target_roll), math.sin(target_yaw) * math.sin(target_pitch) * math.cos(target_roll) - math.cos(target_yaw) * math.sin(target_roll)], [-math.sin(target_pitch), math.cos(target_pitch) * math.sin(target_roll), math.cos(target_pitch) * math.cos(target_roll)]])
    tcp_offset_local = np.array([0, tcp_offset, 0])
    tcp_offset_world = R_target @ tcp_offset_local
    wrist_pos = np.array([x, y, z]) - tcp_offset_world
    wx, wy, wz = wrist_pos
    proj_dist = math.sqrt(wx ** 2 + wz ** 2)
    if proj_dist < 1e-10:
        theta1 = 0
    else:
        theta1 = math.atan2(wx, wz)
    j3_height = 0.39225
    j3_to_wrist = 0.09465
    r_wrist = math.sqrt(wx ** 2 + wz ** 2)
    h_wrist = wy - 0.093
    theta2 = math.atan2(h_wrist, r_wrist) - theta1
    R_theta1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    R_theta2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R_12 = R_theta1 @ R_theta2
    R_remaining = R_12.T @ R_target
    theta3 = math.atan2(R_remaining[0, 1], R_remaining[0, 0])
    R_theta3 = np.array([[math.cos(theta3), -math.sin(theta3), 0], [math.sin(theta3), math.cos(theta3), 0], [0, 0, 1]])
    R_123 = R_12 @ R_theta3
    R_remaining = R_123.T @ R_target
    theta4 = math.atan2(-R_remaining[2, 0], R_remaining[2, 2])
    for angle in [theta1, theta2, theta3, theta4]:
        while angle > 2 * math.pi:
            angle -= 2 * math.pi
        while angle < -2 * math.pi:
            angle += 2 * math.pi
    return (theta1, theta2, theta3, theta4)