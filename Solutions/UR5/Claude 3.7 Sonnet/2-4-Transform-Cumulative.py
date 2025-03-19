def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
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

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    tcp_pos = np.array([x, y, z])
    tcp_offset = np.array([0, 0, 0.09465])
    wrist_pos = tcp_pos - R_target @ tcp_offset
    theta1 = math.atan2(wrist_pos[0], wrist_pos[2])
    joint2_pos = np.array([0, 0, 0.39225])
    R1 = rot_y(theta1)
    wrist_in_j1 = R1.T @ wrist_pos
    j2_to_wrist = wrist_in_j1 - np.array([0, 0, 0.39225])
    dist_j2_to_wrist = np.linalg.norm(j2_to_wrist)
    link2_length = 0.093
    cos_theta2 = j2_to_wrist[2] / dist_j2_to_wrist
    theta2 = math.acos(max(-1, min(1, cos_theta2)))
    if j2_to_wrist[0] < 0:
        theta2 = -theta2
    R2 = rot_y(theta2)
    R_after_j2 = R2.T @ R1.T @ R_target
    theta3 = math.atan2(R_after_j2[1, 0], R_after_j2[0, 0])
    return (theta1, theta2, theta3)