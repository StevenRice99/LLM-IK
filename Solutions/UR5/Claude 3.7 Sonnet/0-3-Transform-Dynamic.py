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
    rx, ry, rz = r
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    tcp_offset_local = np.array([0, 0.093, 0])
    tcp_offset_global = R @ tcp_offset_local
    wrist_x = x - tcp_offset_global[0]
    wrist_y = y - tcp_offset_global[1]
    wrist_z = z - tcp_offset_global[2]
    theta1 = math.atan2(-wrist_x, wrist_y)
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    R1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    wrist_frame1 = R1.T @ np.array([wrist_x, wrist_y, wrist_z])
    wrist_rel_joint2 = wrist_frame1 - np.array([0, 0.13585, 0])
    L2 = math.sqrt(0.1197 ** 2 + 0.425 ** 2)
    L3 = 0.39225
    phi = math.atan2(0.425, -0.1197)
    D = math.sqrt(wrist_rel_joint2[1] ** 2 + wrist_rel_joint2[2] ** 2)
    cos_theta3 = (D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    cos_alpha = (L2 ** 2 + D ** 2 - L3 ** 2) / (2 * L2 * D)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    beta = math.atan2(wrist_rel_joint2[2], wrist_rel_joint2[1])
    theta2 = beta - alpha - phi
    R_desired_frame1 = R1.T @ R
    sin_ry_frame1 = R_desired_frame1[0, 2]
    cos_ry_frame1 = R_desired_frame1[2, 2]
    ry_frame1 = math.atan2(sin_ry_frame1, cos_ry_frame1)
    theta4 = ry_frame1 - theta2 - theta3
    return (theta1, theta2, theta3, theta4)