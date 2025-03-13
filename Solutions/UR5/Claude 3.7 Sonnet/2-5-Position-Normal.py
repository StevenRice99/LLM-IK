def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    import math
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    l4 = 0.0823
    theta4 = 0
    theta1 = np.arctan2(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    h = z
    d = np.sqrt((r - l2) ** 2 + (h - l1) ** 2)
    d_3_tcp = np.sqrt(l3 ** 2 + l4 ** 2)
    phi = np.arctan2(l4, l3)
    cos_theta3 = (l2 ** 2 + d_3_tcp ** 2 - d ** 2) / (2 * l2 * d_3_tcp)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3) - phi
    alpha = np.arctan2(h - l1, r - l2)
    beta = np.arcsin(d_3_tcp * np.sin(theta3 + phi) / d)
    theta2 = alpha - beta
    d_2_tcp = np.sqrt(l2 ** 2 + l3 ** 2 + l4 ** 2)
    h_2 = l1
    r_xy = np.sqrt(x ** 2 + y ** 2)
    h_target = z - h_2
    d_2_target = np.sqrt((r_xy - l2) ** 2 + h_target ** 2)
    if d_2_target > d_2_tcp:
        ratio = d_2_tcp / d_2_target
        r_xy = l2 + (r_xy - l2) * ratio
        h_target = h_target * ratio
    if r_xy > 0:
        ux = x / r_xy
        uy = y / r_xy
    else:
        ux = 0
        uy = 1
    vx = -uy
    vy = ux
    wx = x - l4 * vx
    wy = y - l4 * vy
    wz = z
    wr_xy = np.sqrt(wx ** 2 + wy ** 2)
    theta1 = np.arctan2(wx, wy)
    wr = wr_xy
    wh = wz
    d_2_wrist = np.sqrt((wr - l2) ** 2 + (wh - l1) ** 2)
    cos_theta3 = (l3 ** 2 + l2 ** 2 - d_2_wrist ** 2) / (2 * l3 * l2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    alpha = np.arctan2(wh - l1, wr - l2)
    beta = np.arcsin(l3 * np.sin(-theta3) / d_2_wrist)
    theta2 = alpha + beta
    return (theta1, theta2, theta3, theta4)