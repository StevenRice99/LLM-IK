def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    from math import atan2, sin, cos, sqrt, acos, pi
    l1_y = -0.1197
    l1_z = 0.425
    l2_z = 0.39225
    l3_y = 0.093
    l4_z = 0.09465
    l5_y = 0.0823
    px, py, pz = p
    rx, ry, rz = r

    def rotation_matrix(axis, angle):
        """Create rotation matrix for rotation around given axis."""
        c, s = (cos(angle), sin(angle))
        if axis == 'x':
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == 'z':
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Rx = rotation_matrix('x', rx)
    Ry = rotation_matrix('y', ry)
    Rz = rotation_matrix('z', rz)
    R_target = Rz @ Ry @ Rx
    tcp_rot_offset = rotation_matrix('z', 1.570796325)
    R_wrist = R_target @ tcp_rot_offset.T
    tcp_offset = np.array([0, l5_y, 0])
    tcp_offset_world = R_target @ tcp_offset
    wrist_center = np.array(p) - tcp_offset_world
    wcx, wcy, wcz = wrist_center
    theta1 = atan2(wcx, wcz)
    r_wc = sqrt(wcx ** 2 + wcz ** 2)
    d_wc = wcy
    d_j2 = d_wc - l1_y
    h_j2 = r_wc - l1_z
    d_j2_wc = sqrt(d_j2 ** 2 + h_j2 ** 2)
    l3_eff = sqrt(l3_y ** 2 + l4_z ** 2)
    phi_l3 = atan2(l3_y, l4_z)
    alpha = atan2(d_j2, h_j2)
    cos_beta = (d_j2_wc ** 2 + l2_z ** 2 - l3_eff ** 2) / (2 * d_j2_wc * l2_z)
    cos_beta = min(1, max(-1, cos_beta))
    beta = acos(cos_beta)
    theta2 = alpha - beta
    cos_gamma = (l2_z ** 2 + l3_eff ** 2 - d_j2_wc ** 2) / (2 * l2_z * l3_eff)
    cos_gamma = min(1, max(-1, cos_gamma))
    gamma = acos(cos_gamma)
    theta3 = pi / 2 - (gamma + phi_l3)
    R01 = rotation_matrix('y', theta1)
    R12 = rotation_matrix('y', theta2)
    R23 = rotation_matrix('y', theta3)
    R03 = R01 @ R12 @ R23
    R3w = R03.T @ R_wrist
    s5 = -R3w[2, 0]
    c5 = R3w[2, 2]
    theta5 = atan2(s5, c5)
    if abs(s5) > 0.99999:
        theta4 = 0
    else:
        s4 = R3w[1, 0] / cos(theta5)
        c4 = R3w[0, 0] / cos(theta5)
        theta4 = atan2(s4, c4)

    def normalize_angle(angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle
    theta1 = normalize_angle(theta1)
    theta2 = normalize_angle(theta2)
    theta3 = normalize_angle(theta3)
    theta4 = normalize_angle(theta4)
    theta5 = normalize_angle(theta5)
    return (theta1, theta2, theta3, theta4, theta5)