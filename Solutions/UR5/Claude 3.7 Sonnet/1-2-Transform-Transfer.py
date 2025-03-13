def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    y_offset = -0.1197
    l1 = 0.425
    l2 = 0.39225
    flipped = abs(rx - np.pi) < 1e-10 and abs(rz - np.pi) < 1e-10
    if flipped:
        z = -z
    r_xz = np.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r_xz ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2_pos = np.arccos(cos_theta2)
    theta2_neg = -np.arccos(cos_theta2)
    alpha = np.arctan2(x, z)
    beta_pos = np.arctan2(l2 * np.sin(theta2_pos), l1 + l2 * np.cos(theta2_pos))
    beta_neg = np.arctan2(l2 * np.sin(theta2_neg), l1 + l2 * np.cos(theta2_neg))
    theta1_pos = alpha - beta_pos
    theta1_neg = alpha - beta_neg
    if flipped:
        theta1_pos = theta1_pos + np.pi
        theta1_neg = theta1_neg + np.pi
        if theta1_pos > np.pi:
            theta1_pos = theta1_pos - 2 * np.pi
        elif theta1_pos < -np.pi:
            theta1_pos = theta1_pos + 2 * np.pi
        if theta1_neg > np.pi:
            theta1_neg = theta1_neg - 2 * np.pi
        elif theta1_neg < -np.pi:
            theta1_neg = theta1_neg + 2 * np.pi
    err_pos = abs(theta1_pos + theta2_pos - ry)
    err_neg = abs(theta1_neg + theta2_neg - ry)
    if err_pos <= err_neg:
        return (theta1_pos, theta2_pos)
    else:
        return (theta1_neg, theta2_neg)