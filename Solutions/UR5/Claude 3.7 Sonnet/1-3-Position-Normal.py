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
    x_rot = x * np.cos(-theta1) - z * np.sin(-theta1)
    y_rot = y
    z_rot = x * np.sin(-theta1) + z * np.cos(-theta1)
    y_rot = y_rot - y_offset_tcp
    y_rot = y_rot - y_offset_link2
    L2 = z_offset_link2
    L3 = z_offset_link3
    D = np.sqrt(y_rot ** 2 + z_rot ** 2)
    cos_theta3 = (D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    phi = np.arctan2(y_rot, z_rot)
    cos_alpha = (L2 ** 2 + D ** 2 - L3 ** 2) / (2 * L2 * D)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    theta2 = phi - alpha
    return (theta1, theta2, theta3)