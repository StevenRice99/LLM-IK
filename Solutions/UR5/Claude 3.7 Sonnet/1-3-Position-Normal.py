def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    y_offset_link2 = -0.1197
    z_offset_link2 = 0.425
    z_offset_link3 = 0.39225
    y_offset_tcp = 0.093
    theta1 = np.arctan2(x, z)
    r_xz = np.sqrt(x ** 2 + z ** 2)
    y_adj = y - y_offset_tcp
    y_from_link2 = y_adj - y_offset_link2
    d = np.sqrt(r_xz ** 2 + y_from_link2 ** 2)
    cos_theta3 = (d ** 2 - z_offset_link2 ** 2 - z_offset_link3 ** 2) / (2 * z_offset_link2 * z_offset_link3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.pi - np.arccos(cos_theta3)
    beta = np.arccos((z_offset_link2 ** 2 + d ** 2 - z_offset_link3 ** 2) / (2 * z_offset_link2 * d))
    alpha = np.arctan2(y_from_link2, r_xz)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)