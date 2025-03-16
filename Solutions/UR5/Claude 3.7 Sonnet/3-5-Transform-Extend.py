def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    px, py, pz = p
    rx, ry, rz = r
    l1_y = 0.093
    l2_z = 0.09465
    tcp_y = 0.0823

    def euler_to_rotation_matrix(euler_angles):
        x, y, z = euler_angles
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        return np.matmul(np.matmul(Rz, Ry), Rx)
    R_target = euler_to_rotation_matrix([rx, ry, rz])
    R_tcp_offset = euler_to_rotation_matrix([0, 0, 1.570796325])
    R_tcp_offset_inv = np.transpose(R_tcp_offset)
    R_link3 = np.matmul(R_target, R_tcp_offset_inv)
    tcp_offset = np.array([0, tcp_y, 0])
    tcp_offset_world = np.matmul(R_target, tcp_offset)
    link3_pos = np.array([px, py, pz]) - tcp_offset_world
    joint1 = math.atan2(link3_pos[0], link3_pos[2])
    R_y = lambda angle: np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    R_z = lambda angle: np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R_1 = R_y(joint1)
    link3_in_1 = np.matmul(np.transpose(R_1), link3_pos)
    joint2_in_1 = np.array([0, l1_y, 0])
    v = link3_in_1 - joint2_in_1
    v_xy = np.array([v[0], v[1], 0])
    v_xy_length = np.linalg.norm(v_xy)
    if v_xy_length < 1e-06:
        joint2 = 0
    else:
        v_xy_norm = v_xy / v_xy_length
        y_axis = np.array([0, 1, 0])
        dot_product = np.dot(v_xy_norm, y_axis)
        angle = math.acos(np.clip(dot_product, -1.0, 1.0))
        if v_xy_norm[0] < 0:
            angle = -angle
        joint2 = angle
    R_2 = R_z(joint2)
    R_12 = np.matmul(R_1, R_2)
    R_12_inv = np.transpose(R_12)
    R_3_needed = np.matmul(R_12_inv, R_link3)
    cos_theta = R_3_needed[2, 2]
    sin_theta = -R_3_needed[0, 2]
    joint3 = math.atan2(sin_theta, cos_theta)
    return (joint1, joint2, joint3)