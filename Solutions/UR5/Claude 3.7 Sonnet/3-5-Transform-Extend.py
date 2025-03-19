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

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    tcp_offset = rot_z(1.570796325)
    R_target_adjusted = R_target @ tcp_offset.T
    joint1 = math.atan2(x, z)
    R_joint1 = rot_y(joint1)
    R_remaining = R_joint1.T @ R_target_adjusted
    joint2 = math.atan2(R_remaining[1, 0], R_remaining[0, 0])
    R_joint2 = rot_z(joint2)
    R_final = R_joint2.T @ R_remaining
    joint3 = math.atan2(-R_final[2, 0], R_final[0, 0])
    return (joint1, joint2, joint3)