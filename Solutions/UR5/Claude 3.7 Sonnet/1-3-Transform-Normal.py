def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    _, pitch, _ = r
    l1 = 0.425
    l2 = 0.39225
    d1 = -0.1197
    d3 = 0.093
    joint1 = np.arctan2(x, z)
    r_xz = np.sqrt(x ** 2 + z ** 2)
    ee_orientation = pitch
    wrist_x = x - d3 * np.sin(joint1) * np.sin(ee_orientation)
    wrist_y = y - d3 * np.cos(ee_orientation)
    wrist_z = z - d3 * np.cos(joint1) * np.sin(ee_orientation)
    wrist_r_xz = np.sqrt(wrist_x ** 2 + wrist_z ** 2)
    r_planar = np.sqrt(wrist_r_xz ** 2 + (wrist_y - d1) ** 2)
    cos_joint3 = (r_planar ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_joint3 = np.clip(cos_joint3, -1.0, 1.0)
    joint3 = -np.arccos(cos_joint3)
    beta = np.arctan2(wrist_y - d1, wrist_r_xz)
    gamma = np.arctan2(l2 * np.sin(-joint3), l1 + l2 * np.cos(-joint3))
    joint2 = beta - gamma
    joint3_adjusted = ee_orientation - joint1 - joint2
    return (joint1, joint2, joint3_adjusted)