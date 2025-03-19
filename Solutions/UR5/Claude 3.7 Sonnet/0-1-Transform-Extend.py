def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
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
    R = Rz @ Ry @ Rx
    tcp_offset = np.array([0, -0.1197, 0.425])
    j2_pos = np.array([x, y, z]) - R @ tcp_offset
    theta1 = math.atan2(j2_pos[0], j2_pos[1])
    l1 = 0.13585
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    j2_x1 = c1 * j2_pos[0] + s1 * j2_pos[1]
    j2_y1 = -s1 * j2_pos[0] + c1 * j2_pos[1]
    j2_z1 = j2_pos[2]
    if abs(j2_x1) > 1e-10 or abs(j2_y1 - l1) > 1e-10 or abs(j2_z1) > 1e-10:
        R1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        R2 = R1.T @ R
        theta2 = math.atan2(R2[0, 2], R2[0, 0])
    else:
        R1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        R2 = R1.T @ R
        theta2 = math.atan2(R2[0, 2], R2[0, 0])
    return (theta1, theta2)