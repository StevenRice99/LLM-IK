def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    px, py, pz = p
    rx, ry, rz = r
    l1_y = 0.093
    l2_z = 0.09465
    tcp_y = 0.0823

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    tcp_offset = rot_z(1.570796325)
    R_target_adjusted = R_target @ np.linalg.inv(tcp_offset)
    joint1 = math.atan2(px, pz)
    R1 = rot_y(joint1)
    R1_inv = np.linalg.inv(R1)
    R_after_joint1 = R1_inv @ R_target_adjusted
    joint2 = math.atan2(R_after_joint1[0, 1], R_after_joint1[0, 0])
    R2 = rot_z(joint2)
    R2_inv = np.linalg.inv(R2)
    R_after_joint2 = R2_inv @ R_after_joint1
    joint3 = math.atan2(-R_after_joint2[2, 0], R_after_joint2[2, 2])
    return (joint1, joint2, joint3)