def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    y_offset_joint2 = -0.1197
    y_offset_joint4 = 0.093
    z_offset_tcp = 0.09465
    l1 = 0.425
    l2 = 0.39225
    joint1 = np.arctan2(x, z)
    joint4 = 0.0
    r_xz = np.sqrt(x ** 2 + z ** 2)
    x_joint4 = x - z_offset_tcp * np.sin(joint1)
    z_joint4 = z - z_offset_tcp * np.cos(joint1)
    r_xz_joint4 = np.sqrt(x_joint4 ** 2 + z_joint4 ** 2)
    y_joint4 = y - y_offset_joint4
    y_adjusted = y_joint4 - y_offset_joint2
    r = np.sqrt(r_xz_joint4 ** 2 + y_adjusted ** 2)
    cos_joint3 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_joint3 = np.clip(cos_joint3, -1.0, 1.0)
    joint3 = np.arccos(cos_joint3)
    beta = np.arctan2(y_adjusted, r_xz_joint4)
    alpha = np.arctan2(l2 * np.sin(joint3), l1 + l2 * np.cos(joint3))
    joint2 = beta - alpha
    return (joint1, joint2, joint3, joint4)