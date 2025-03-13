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
    roll, pitch, yaw = r
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    c3 = np.cos(yaw)
    s3 = np.sin(yaw)
    R3 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    tcp_offset = np.array([0, 0, l3])
    j3_pos = np.array([x, y, z]) - R3 @ tcp_offset
    j2_pos = np.array([j3_pos[0], j3_pos[1] - l2, j3_pos[2]])
    joint1 = np.arctan2(j2_pos[0], j2_pos[2])

    def Rx(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def Ry(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def Rz(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    R1 = Ry(joint1)
    R2_desired = np.linalg.inv(R1) @ R_target @ np.linalg.inv(Rz(yaw))
    c2 = R2_desired[0, 0]
    s2 = -R2_desired[2, 0]
    joint2 = np.arctan2(s2, c2)
    joint3 = yaw
    joint1 = (joint1 + np.pi) % (2 * np.pi) - np.pi
    joint2 = (joint2 + np.pi) % (2 * np.pi) - np.pi
    joint3 = (joint3 + np.pi) % (2 * np.pi) - np.pi
    return (joint1, joint2, joint3)