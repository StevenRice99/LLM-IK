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
    l2 = 0.39225
    l3_offset_y = 0.093
    tcp_offset_z = 0.09465
    R_target = np.array([[math.cos(ry) * math.cos(rz), -math.cos(ry) * math.sin(rz), math.sin(ry)], [math.cos(rx) * math.sin(rz) + math.sin(rx) * math.sin(ry) * math.cos(rz), math.cos(rx) * math.cos(rz) - math.sin(rx) * math.sin(ry) * math.sin(rz), -math.sin(rx) * math.cos(ry)], [math.sin(rx) * math.sin(rz) - math.cos(rx) * math.sin(ry) * math.cos(rz), math.sin(rx) * math.cos(rz) + math.cos(rx) * math.sin(ry) * math.sin(rz), math.cos(rx) * math.cos(ry)]])
    local_tcp_offset = np.array([0, 0, tcp_offset_z])
    world_tcp_offset = R_target @ local_tcp_offset
    j3_pos = np.array([x, y, z]) - world_tcp_offset
    theta1 = math.atan2(j3_pos[0], j3_pos[2])
    if abs(rx - math.pi) < 1e-10 and abs(rz - math.pi) < 1e-10:
        theta2 = -ry - theta1 + math.pi
        while theta2 > 2 * math.pi:
            theta2 -= 2 * math.pi
        while theta2 < -2 * math.pi:
            theta2 += 2 * math.pi
    else:
        theta2 = ry - theta1
    R1 = np.array([[math.cos(theta1), 0, math.sin(theta1)], [0, 1, 0], [-math.sin(theta1), 0, math.cos(theta1)]])
    R2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R12 = R2 @ R1
    R3_needed = np.transpose(R12) @ R_target
    theta3 = math.atan2(R3_needed[1, 0], R3_needed[0, 0])
    return (theta1, theta2, theta3)