def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    px, py, pz = p
    rx, ry, rz = r
    l1 = 0.13585
    l2_y = -0.1197
    l2_z = 0.425
    l3 = 0.39225
    l4 = 0.093
    theta1 = np.arctan2(py, px)
    r_distance = np.sqrt(px ** 2 + py ** 2)
    wrist_y = r_distance - l4 * np.cos(ry)
    wrist_z = pz - l4 * np.sin(ry)
    wrist_to_j2_y = wrist_y - l1
    wrist_to_j2_distance = np.sqrt(wrist_to_j2_y ** 2 + wrist_z ** 2)
    l2 = np.sqrt(l2_y ** 2 + l2_z ** 2)
    cos_theta3 = (l2 ** 2 + l3 ** 2 - wrist_to_j2_distance ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    psi = np.arctan2(wrist_z, wrist_to_j2_y)
    phi = np.arccos((l2 ** 2 + wrist_to_j2_distance ** 2 - l3 ** 2) / (2 * l2 * wrist_to_j2_distance))
    offset_angle = np.arctan2(l2_z, -l2_y)
    theta2 = psi - phi - offset_angle
    theta4 = ry - theta2 - theta3
    return (theta1, theta2, theta3, theta4)