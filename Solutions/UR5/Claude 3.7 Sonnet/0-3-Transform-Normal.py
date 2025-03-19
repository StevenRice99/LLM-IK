def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    l4 = 0.093
    theta1 = np.arctan2(y, x)
    cos_rz, sin_rz = (np.cos(rz), np.sin(rz))
    cos_ry, sin_ry = (np.cos(ry), np.sin(ry))
    cos_rx, sin_rx = (np.cos(rx), np.sin(rx))
    R = np.array([[cos_ry * cos_rz, -cos_rx * sin_rz + sin_rx * sin_ry * cos_rz, sin_rx * sin_rz + cos_rx * sin_ry * cos_rz], [cos_ry * sin_rz, cos_rx * cos_rz + sin_rx * sin_ry * sin_rz, -sin_rx * cos_rz + cos_rx * sin_ry * sin_rz], [-sin_ry, sin_rx * cos_ry, cos_rx * cos_ry]])
    local_offset = np.array([0, l4, 0])
    world_offset = R @ local_offset
    wx = x - world_offset[0]
    wy = y - world_offset[1]
    wz = z - world_offset[2]
    c1, s1 = (np.cos(theta1), np.sin(theta1))
    xw_in_plane = c1 * wx + s1 * wy
    yw_in_plane = -s1 * wx + c1 * wy
    zw_in_plane = wz
    zw_adj = zw_in_plane - l1
    d = np.sqrt(xw_in_plane ** 2 + zw_adj ** 2)
    cos_alpha = (l2 ** 2 + d ** 2 - l3 ** 2) / (2 * l2 * d)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    beta = np.arctan2(zw_adj, xw_in_plane)
    theta2 = beta - alpha
    cos_gamma = (l2 ** 2 + l3 ** 2 - d ** 2) / (2 * l2 * l3)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    theta3 = np.pi - gamma
    R1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    local_R = R1.T @ R
    local_ry = np.arctan2(-local_R[0, 2], local_R[2, 2])
    theta4 = local_ry - theta2 - theta3
    return (theta1, theta2, theta3, theta4)