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
    l1 = 0.425
    l2 = 0.39225
    d1 = -0.1197
    d2 = 0
    d3 = 0.093
    theta1 = math.atan2(x, z)
    tcp_offset = np.array([0, d3, 0])
    R_tcp = np.array([[math.cos(rx) * math.cos(rz) - math.sin(rx) * math.sin(ry) * math.sin(rz), -math.cos(rx) * math.sin(rz) - math.sin(rx) * math.sin(ry) * math.cos(rz), -math.sin(rx) * math.cos(ry)], [math.cos(ry) * math.sin(rz), math.cos(ry) * math.cos(rz), -math.sin(ry)], [math.sin(rx) * math.cos(rz) + math.cos(rx) * math.sin(ry) * math.sin(rz), -math.sin(rx) * math.sin(rz) + math.cos(rx) * math.sin(ry) * math.cos(rz), math.cos(rx) * math.cos(ry)]])
    joint3_to_tcp_global = np.array([0, d3, 0])
    if abs(rx) < 1e-10 and abs(rz) < 1e-10:
        c = math.cos(ry)
        s = math.sin(ry)
        joint3_to_tcp_global = np.array([s * 0, d3, c * 0])
    joint3_pos = np.array([x, y, z]) - joint3_to_tcp_global
    c1 = math.cos(-theta1)
    s1 = math.sin(-theta1)
    x_j1 = c1 * joint3_pos[0] + s1 * joint3_pos[2]
    y_j1 = joint3_pos[1]
    z_j1 = -s1 * joint3_pos[0] + c1 * joint3_pos[2]
    y_rel = y_j1 - d1
    z_rel = z_j1 - l1
    r_j2 = math.sqrt(y_rel ** 2 + z_rel ** 2)
    if r_j2 > l2:
        r_j2 = l2
    phi = math.atan2(y_rel, z_rel)
    theta2 = phi
    theta3 = ry - theta1 - theta2
    while theta1 > 2 * math.pi:
        theta1 -= 2 * math.pi
    while theta1 < -2 * math.pi:
        theta1 += 2 * math.pi
    while theta2 > 2 * math.pi:
        theta2 -= 2 * math.pi
    while theta2 < -2 * math.pi:
        theta2 += 2 * math.pi
    while theta3 > 2 * math.pi:
        theta3 -= 2 * math.pi
    while theta3 < -2 * math.pi:
        theta3 += 2 * math.pi
    return (theta1, theta2, theta3)