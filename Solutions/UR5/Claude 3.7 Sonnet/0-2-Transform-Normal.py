def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.13585
    l2 = 0.425
    d2 = 0.1197
    l3 = 0.39225
    theta1 = np.arctan2(y, x)
    r_xy = np.sqrt(x ** 2 + y ** 2)
    r_target = np.sqrt(x ** 2 + y ** 2)
    z_adj = z
    l2_eff = np.sqrt(l2 ** 2 + d2 ** 2)
    d = np.sqrt(r_target ** 2 + (z_adj - l1) ** 2)
    cos_theta3 = (d ** 2 - l2_eff ** 2 - l3 ** 2) / (2 * l2_eff * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    theta3 = -theta3
    phi = np.arctan2(z_adj - l1, r_target)
    sin_alpha = l3 * np.sin(abs(theta3)) / d
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    alpha = np.arcsin(sin_alpha)
    theta2 = phi - alpha
    return (theta1, theta2, theta3)