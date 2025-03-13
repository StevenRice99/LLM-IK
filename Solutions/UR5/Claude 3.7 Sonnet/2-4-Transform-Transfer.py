def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465

    def Rx(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def Ry(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def Rz(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = Rz(rz) @ Ry(ry) @ Rx(rx)
    tcp_offset_local = np.array([0, 0, l3])
    tcp_offset_global = R_target @ tcp_offset_local
    j3_pos = np.array([x, y, z]) - tcp_offset_global
    j3_x, j3_y, j3_z = j3_pos
    theta3 = np.arctan2(R_target[1, 0], R_target[0, 0])
    r_xz = np.sqrt(j3_x ** 2 + j3_z ** 2)
    cos_theta2 = (r_xz ** 2 - l1 ** 2) / (2 * l1 * r_xz)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    theta1 = np.arctan2(j3_x, j3_z)
    R_solution = Ry(theta1) @ Ry(theta2) @ Rz(theta3)
    R_without_theta3 = R_target @ Rz(-theta3)
    sin_theta2 = R_without_theta3[0, 2]
    sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
    theta2 = np.arcsin(sin_theta2)
    cos_theta1 = R_without_theta3[2, 2] / np.cos(theta2)
    sin_theta1 = -R_without_theta3[1, 2] / np.cos(theta2)
    theta1 = np.arctan2(sin_theta1, cos_theta1)
    return (theta1, theta2, theta3)