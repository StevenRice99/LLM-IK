def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    from math import atan2, acos, sqrt, sin, cos, pi
    x, y, z = p
    roll, pitch, yaw = r
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    l4 = 0.0823

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, cos(angle), -sin(angle)], [0, sin(angle), cos(angle)]])

    def rot_y(angle):
        return np.array([[cos(angle), 0, sin(angle)], [0, 1, 0], [-sin(angle), 0, cos(angle)]])

    def rot_z(angle):
        return np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])

    def transform(rot, trans):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    T_target = transform(R_target, [x, y, z])
    T_tcp = transform(rot_z(pi / 2), [0, l4, 0])
    T_wrist = T_target @ np.linalg.inv(T_tcp)
    wx, wy, wz = T_wrist[:3, 3]
    theta1 = atan2(wx, wz)
    c1, s1 = (cos(theta1), sin(theta1))
    R1 = np.array([[c1, 0, s1], [0, 1, 0], [-s1, 0, c1]])
    wrist_pos_1 = R1.T @ np.array([wx, wy, wz])
    wx1, wy1, wz1 = wrist_pos_1
    dx = wx1
    dy = wy1 - l2
    dz = wz1 - l1
    d = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    cos_theta2 = (l3 ** 2 + l1 ** 2 - d ** 2) / (2 * l3 * l1)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = -acos(cos_theta2)
    alpha = atan2(dz, sqrt(dx ** 2 + dy ** 2))
    theta2 -= alpha
    R1 = rot_y(theta1)
    R2 = rot_y(theta2)
    R12 = R1 @ R2
    R_desired = R_target @ rot_z(-pi / 2)
    R3_needed = np.linalg.inv(R12) @ R_desired
    theta3 = atan2(R3_needed[1, 0], R3_needed[0, 0])
    R3 = rot_z(theta3)
    R123 = R12 @ R3
    R4_needed = np.linalg.inv(R123) @ R_desired
    theta4 = atan2(-R4_needed[2, 0], R4_needed[0, 0])
    return (theta1, theta2, theta3, theta4)