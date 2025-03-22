def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    l1_y = 0.13585
    l2_y = -0.1197
    l2_z = 0.425
    l3_z = 0.39225
    tcp_y = 0.093
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = yaw
    c1, s1 = (np.cos(theta1), np.sin(theta1))
    R1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    p1 = R1 @ np.array([x, y, z])
    c_r, s_r = (np.cos(roll), np.sin(roll))
    c_p, s_p = (np.cos(pitch), np.sin(pitch))
    c_y, s_y = (np.cos(yaw), np.sin(yaw))
    Rx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    Ry = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    Rz = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
    R_target = Rz @ Ry @ Rx
    R1_target = R1 @ R_target
    tcp_offset = R1_target @ np.array([0, tcp_y, 0])
    wrist = p1 - tcp_offset
    wrist[1] -= l1_y
    l2 = np.sqrt(l2_y ** 2 + l2_z ** 2)
    l3 = l3_z
    phi2 = np.arctan2(l2_z, l2_y)
    d = np.sqrt(wrist[1] ** 2 + wrist[2] ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    theta3_alt = -theta3
    beta = np.arctan2(wrist[1], wrist[2])
    gamma = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = beta - gamma
    gamma_alt = np.arctan2(l3 * np.sin(theta3_alt), l2 + l3 * np.cos(theta3_alt))
    theta2_alt = beta - gamma_alt
    z_axis = R1_target[:, 2]
    target_angle = np.arctan2(z_axis[1], z_axis[2])
    theta4 = target_angle - theta2 - theta3
    theta4_alt = target_angle - theta2_alt - theta3_alt

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    theta2 = normalize_angle(theta2 - phi2)
    theta2_alt = normalize_angle(theta2_alt - phi2)
    theta1 = normalize_angle(theta1)
    theta3 = normalize_angle(theta3)
    theta4 = normalize_angle(theta4)
    theta3_alt = normalize_angle(theta3_alt)
    theta4_alt = normalize_angle(theta4_alt)
    solution1 = (theta1, theta2, theta3, theta4)
    solution2 = (theta1, theta2_alt, theta3_alt, theta4_alt)
    if sum((abs(angle) for angle in solution1)) <= sum((abs(angle) for angle in solution2)):
        return solution1
    else:
        return solution2