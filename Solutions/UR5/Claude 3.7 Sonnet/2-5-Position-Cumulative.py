def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.39225
    l2 = 0.09465
    y_offset_j3 = 0.093
    y_offset_tcp = 0.0823
    y_target = y - y_offset_tcp
    if abs(x) < 1e-10 and abs(z) < 1e-10:
        joint3 = 0.0
    else:
        joint3 = np.arctan2(x, z)
    x_rot = x * np.cos(-joint3) - z * np.sin(-joint3)
    z_rot = x * np.sin(-joint3) + z * np.cos(-joint3)
    r = np.sqrt(x_rot ** 2 + z_rot ** 2)
    y_adj = y_target - y_offset_j3
    d = np.sqrt(r ** 2 + y_adj ** 2)
    cos_joint4 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_joint4 = np.clip(cos_joint4, -1.0, 1.0)
    joint4 = np.arccos(cos_joint4)
    alpha = np.arctan2(y_adj, r)
    beta = np.arctan2(l2 * np.sin(joint4), l1 + l2 * np.cos(joint4))
    joint1 = alpha - beta
    joint2 = 0.0
    return (joint1, joint2, joint3, joint4)