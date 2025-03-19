def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.39225
    l2_y = 0.093
    l3_z = 0.09465

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    tcp_offset_world = R_target @ np.array([0, 0, l3_z])
    j3_pos_world = np.array([x, y, z]) - tcp_offset_world
    joint1 = math.atan2(j3_pos_world[0], j3_pos_world[2])
    R1 = rot_y(joint1)
    j3_pos_j1 = R1.T @ j3_pos_world
    R1_to_target = R1.T @ R_target
    j2_pos_j1 = np.array([0, 0, l1])
    j3_from_j2_j1 = j3_pos_j1 - j2_pos_j1
    j3_from_j2_j1_xz = np.array([j3_from_j2_j1[0], 0, j3_from_j2_j1[2]])
    xz_length = np.linalg.norm(j3_from_j2_j1_xz)
    basic_angle = math.atan2(j3_from_j2_j1[0], j3_from_j2_j1[2])
    joint2_config1 = basic_angle
    joint2_config2 = basic_angle + math.pi
    joint2_config3 = basic_angle - math.pi
    R2_config1 = rot_y(joint2_config1)
    R2_config2 = rot_y(joint2_config2)
    R2_config3 = rot_y(joint2_config3)
    R12_config1 = R1 @ R2_config1
    R12_config2 = R1 @ R2_config2
    R12_config3 = R1 @ R2_config3
    R2_to_target_config1 = R2_config1.T @ R1_to_target
    R2_to_target_config2 = R2_config2.T @ R1_to_target
    R2_to_target_config3 = R2_config3.T @ R1_to_target
    joint3_config1 = math.atan2(R2_to_target_config1[1, 0], R2_to_target_config1[0, 0])
    joint3_config2 = math.atan2(R2_to_target_config2[1, 0], R2_to_target_config2[0, 0])
    joint3_config3 = math.atan2(R2_to_target_config3[1, 0], R2_to_target_config3[0, 0])
    R_achieved_config1 = R12_config1 @ rot_z(joint3_config1)
    R_achieved_config2 = R12_config2 @ rot_z(joint3_config2)
    R_achieved_config3 = R12_config3 @ rot_z(joint3_config3)
    error_config1 = np.linalg.norm(R_achieved_config1 - R_target, 'fro')
    error_config2 = np.linalg.norm(R_achieved_config2 - R_target, 'fro')
    error_config3 = np.linalg.norm(R_achieved_config3 - R_target, 'fro')
    if error_config1 <= error_config2 and error_config1 <= error_config3:
        joint2 = joint2_config1
        joint3 = joint3_config1
    elif error_config2 <= error_config1 and error_config2 <= error_config3:
        joint2 = joint2_config2
        joint3 = joint3_config2
    else:
        joint2 = joint2_config3
        joint3 = joint3_config3
    return (joint1, joint2, joint3)