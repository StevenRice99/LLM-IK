def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_target = R_z @ R_y @ R_x
    T_target = np.eye(4)
    T_target[:3, :3] = R_target
    T_target[:3, 3] = [x_target, y_target, z_target]
    TCP_offset = np.array([0, 0.0823, 0])
    TCP_rotation_z = 1.570796325
    R_tcp_offset = np.array([[math.cos(TCP_rotation_z), -math.sin(TCP_rotation_z), 0], [math.sin(TCP_rotation_z), math.cos(TCP_rotation_z), 0], [0, 0, 1]])
    R_5 = R_target @ R_tcp_offset.T
    TCP_offset_global = R_target @ TCP_offset
    wrist_pos = np.array([x_target - TCP_offset_global[0], y_target - TCP_offset_global[1], z_target - TCP_offset_global[2]])
    wrist_xy_dist = math.sqrt(wrist_pos[0] ** 2 + wrist_pos[1] ** 2)
    if wrist_xy_dist < 1e-06:
        theta1 = math.atan2(R_target[1, 0], R_target[0, 0])
    else:
        theta1 = math.atan2(wrist_pos[1], wrist_pos[0])
    R_1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    wrist_in_j1 = R_1.T @ wrist_pos
    j5_offset = 0.09465
    j4_pos_in_j1 = np.array([wrist_in_j1[0], wrist_in_j1[1], wrist_in_j1[2] - j5_offset])
    j4_offset = 0.093
    j3_pos_in_j1 = np.array([j4_pos_in_j1[0], j4_pos_in_j1[1] - j4_offset, j4_pos_in_j1[2]])
    a1 = 0.425
    a2 = 0.39225
    j2_offset = -0.1197
    j3_pos_adjusted = np.array([j3_pos_in_j1[0], j3_pos_in_j1[1] - j2_offset, j3_pos_in_j1[2]])
    r = math.sqrt(j3_pos_adjusted[0] ** 2 + j3_pos_adjusted[2] ** 2)
    cos_theta2 = (r ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_pos = math.acos(cos_theta2)
    theta2_neg = -theta2_pos

    def calc_theta1_2(theta2):
        cos_theta1_2 = (a1 + a2 * math.cos(theta2)) / r
        sin_theta1_2 = a2 * math.sin(theta2) / r
        return math.atan2(sin_theta1_2, cos_theta1_2)
    theta1_2_pos = calc_theta1_2(theta2_pos)
    theta1_2_neg = calc_theta1_2(theta2_neg)
    phi = math.atan2(j3_pos_adjusted[0], j3_pos_adjusted[2])
    theta1_prime_pos = phi - theta1_2_pos
    theta1_prime_neg = phi - theta1_2_neg
    theta2_solution1 = theta1_prime_pos
    theta2_solution2 = theta1_prime_neg
    theta3_solution1 = theta2_pos
    theta3_solution2 = theta2_neg
    theta2 = theta2_solution1
    theta3 = theta3_solution1
    R_2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R_3 = np.array([[math.cos(theta3), 0, math.sin(theta3)], [0, 1, 0], [-math.sin(theta3), 0, math.cos(theta3)]])
    R_03 = R_1 @ R_2 @ R_3
    R_35 = R_03.T @ R_5
    theta4 = math.atan2(R_35[1, 0], R_35[0, 0])
    R_4 = np.array([[math.cos(theta4), -math.sin(theta4), 0], [math.sin(theta4), math.cos(theta4), 0], [0, 0, 1]])
    R_45 = R_4.T @ R_35
    theta5 = math.atan2(R_45[0, 2], R_45[2, 2])
    return (theta1, theta2, theta3, theta4, theta5)