def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.093
    L5 = 0.09465
    L6 = 0.0823
    Y1 = 0.13585
    Y2 = -0.1197
    Y3 = 0
    Y4 = 0.093
    Y5 = 0
    Y6 = 0.0823

    def rotx(theta):
        return np.array([[1, 0, 0], [0, math.cos(theta), -math.sin(theta)], [0, math.sin(theta), math.cos(theta)]])

    def roty(theta):
        return np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])

    def rotz(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    rx, ry, rz = r
    R_target = rotz(rz) @ roty(ry) @ rotx(rx)
    R_TCP = rotz(math.pi / 2)
    R_6 = R_target @ R_TCP.T
    tcp_offset = np.array([0, L6, 0])
    offset_world = R_target @ tcp_offset
    wc = np.array(p) - offset_world
    q1 = math.atan2(wc[1], wc[0])
    R_1 = rotz(q1)
    wc_1 = R_1.T @ wc
    wc_2 = [wc_1[0], wc_1[1] - Y1, wc_1[2] - L1]
    y_offset = Y2 + Y4
    wc_2[1] -= y_offset
    r = math.sqrt(wc_2[0] ** 2 + wc_2[1] ** 2 + wc_2[2] ** 2)
    cos_q3 = (r ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_pos = math.acos(cos_q3)
    q3_neg = -q3_pos
    q3 = q3_neg
    r_proj = math.sqrt(wc_2[0] ** 2 + wc_2[2] ** 2)
    gamma = math.atan2(wc_2[0], wc_2[2])
    delta = math.atan2(L3 * math.sin(q3), L2 + L3 * math.cos(q3))
    q2 = gamma - delta
    R_2 = roty(q2)
    R_3 = roty(q3)
    R_0_3 = R_1 @ R_2 @ R_3
    R_3_6 = R_0_3.T @ R_6
    q5 = math.atan2(math.sqrt(R_3_6[0, 0] ** 2 + R_3_6[2, 0] ** 2), R_3_6[1, 0])
    if abs(math.sin(q5)) < 1e-06:
        q4 = 0
        q6 = math.atan2(R_3_6[2, 1], R_3_6[0, 1])
    else:
        q4 = math.atan2(-R_3_6[1, 2], R_3_6[1, 1])
        q6 = math.atan2(-R_3_6[2, 0], R_3_6[0, 0])

    def distance(pos1, pos2):
        return sum(((a - b) ** 2 for a, b in zip(pos1, pos2)))
    q_neg = (q1, q2, q3, q4, q5, q6)
    q3_alt = q3_pos
    delta_alt = math.atan2(L3 * math.sin(q3_alt), L2 + L3 * math.cos(q3_alt))
    q2_alt = gamma - delta_alt
    R_0_3_alt = R_1 @ roty(q2_alt) @ roty(q3_alt)
    R_3_6_alt = R_0_3_alt.T @ R_6
    q5_alt = math.atan2(math.sqrt(R_3_6_alt[0, 0] ** 2 + R_3_6_alt[2, 0] ** 2), R_3_6_alt[1, 0])
    if abs(math.sin(q5_alt)) < 1e-06:
        q4_alt = 0
        q6_alt = math.atan2(R_3_6_alt[2, 1], R_3_6_alt[0, 1])
    else:
        q4_alt = math.atan2(-R_3_6_alt[1, 2], R_3_6_alt[1, 1])
        q6_alt = math.atan2(-R_3_6_alt[2, 0], R_3_6_alt[0, 0])
    q_pos = (q1, q2_alt, q3_alt, q4_alt, q5_alt, q6_alt)
    if abs(q3) < abs(q3_alt):
        return q_neg
    else:
        return q_pos