def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x_target, y_target, z_target = p
    rx, ry, rz = r
    l1_z = 0.13585
    l2 = 0.425
    l3_x = 0
    l3_y = -0.1197
    l3_z = 0.425
    tcp_z = 0.39225
    theta1 = rz
    R1 = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    target_pos = np.array([x_target, y_target, z_target])
    pos_in_frame1 = R1.T @ target_pos
    x1 = pos_in_frame1[0]
    y1 = pos_in_frame1[1]
    z1 = pos_in_frame1[2]
    cx, sx = (math.cos(rx), math.sin(rx))
    cy, sy = (math.cos(ry), math.sin(ry))
    cz, sz = (math.cos(rz), math.sin(rz))
    R_target = np.array([[cy * cz, cy * sz, -sy], [-cx * sz + sx * sy * cz, cx * cz + sx * sy * sz, sx * cy], [sx * sz + cx * sy * cz, -sx * cz + cx * sy * sz, cx * cy]])
    R_target_in_frame1 = R1.T @ R_target
    tcp_offset = tcp_z * R_target_in_frame1[:, 2]
    joint3_pos = pos_in_frame1 - tcp_offset
    joint3_rel_to_joint2 = np.array([joint3_pos[0], joint3_pos[1] - l1_z, joint3_pos[2]])
    joint3_distance = np.linalg.norm(joint3_rel_to_joint2)
    cos_theta3 = (joint3_distance ** 2 - l2 ** 2 - (l3_y ** 2 + l3_z ** 2)) / (2 * l2 * math.sqrt(l3_y ** 2 + l3_z ** 2))
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = -math.acos(cos_theta3)
    theta_to_joint3 = math.atan2(joint3_rel_to_joint2[0], joint3_rel_to_joint2[2])
    beta = math.atan2(l3_y, l3_z)
    theta2 = theta_to_joint3 - math.atan2(l2 * math.sin(theta3), l2 * math.cos(theta3) + math.sqrt(l3_y ** 2 + l3_z ** 2))
    alt_theta3 = -theta3
    alt_theta2 = theta_to_joint3 - math.atan2(l2 * math.sin(alt_theta3), l2 * math.cos(alt_theta3) + math.sqrt(l3_y ** 2 + l3_z ** 2))
    sol1_error = abs(math.sin(theta2 + theta3) - math.sin(ry))
    sol2_error = abs(math.sin(alt_theta2 + alt_theta3) - math.sin(ry))
    if sol2_error < sol1_error:
        theta2, theta3 = (alt_theta2, alt_theta3)
    return (theta1, theta2, theta3)