import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    tcp_offset = 0.093
    cx, cy, cz = (np.cos(rx), np.cos(ry), np.cos(rz))
    sx, sy, sz = (np.sin(rx), np.sin(ry), np.sin(rz))
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x
    direction = R @ np.array([0, tcp_offset, 0])
    wx = x - direction[0]
    wy = y - direction[1]
    wz = z - direction[2]
    numerator = wx ** 2 + wy ** 2 + wz ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    q3_alt = -q3
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    xy_dist = np.sqrt(wx ** 2 + wy ** 2)
    if xy_dist < 1e-06:
        q1 = np.arctan2(0, 1)
        if wz >= 0:
            q2 = -np.pi / 2
        else:
            q2 = np.pi / 2
    else:
        S_squared = wx ** 2 + wy ** 2
        S = np.sqrt(S_squared)
        C = wz
        numerator_q2 = S * A - C * B
        denominator_q2 = S * B + C * A
        q2 = np.arctan2(numerator_q2, denominator_q2)
        q1 = np.arctan2(wy, wx)
    R1 = np.array([[np.cos(q1), -np.sin(q1), 0], [np.sin(q1), np.cos(q1), 0], [0, 0, 1]])
    R2 = np.array([[np.cos(q2), 0, np.sin(q2)], [0, 1, 0], [-np.sin(q2), 0, np.cos(q2)]])
    R3 = np.array([[np.cos(q3), 0, np.sin(q3)], [0, 1, 0], [-np.sin(q3), 0, np.cos(q3)]])
    R123 = R1 @ R2 @ R3
    Rd = R
    R4_needed = R123.T @ Rd
    q4 = np.arctan2(-R4_needed[2, 0], R4_needed[0, 0])
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    return (q1, q2, q3, q4)