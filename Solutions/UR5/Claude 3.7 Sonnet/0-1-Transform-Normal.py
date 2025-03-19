def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.13585
    l2_z = 0.425
    l2_y = -0.1197
    joint1 = np.arctan2(y, x)
    x_rot = x * np.cos(-joint1) - y * np.sin(-joint1)
    z_rot = z
    phi = np.arctan2(-l2_y, l2_z)
    d = np.sqrt((x_rot - 0) ** 2 + (z_rot - l1) ** 2)
    cos_alpha = (l2_z ** 2 + l2_y ** 2 + d ** 2 - x_rot ** 2 - (z_rot - l1) ** 2) / (2 * np.sqrt(l2_z ** 2 + l2_y ** 2) * d)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    beta = np.arctan2(x_rot, z_rot - l1)
    joint2 = beta - alpha + phi
    rx, ry, rz = r
    if abs(abs(rx) - np.pi) < 0.1:
        joint2 = beta + alpha + phi
    joint1 = (joint1 + np.pi) % (2 * np.pi) - np.pi
    joint2 = (joint2 + np.pi) % (2 * np.pi) - np.pi
    return (joint1, joint2)