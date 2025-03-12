def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    y_offset1 = 0.13585
    l1 = 0.425
    l2 = 0.39225
    theta1 = np.arctan2(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z_adj = z - y_offset1
    d_squared = r ** 2 + z_adj ** 2
    d = np.sqrt(d_squared)
    cos_alpha = (l1 ** 2 + d_squared - l2 ** 2) / (2 * l1 * d)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    cos_beta = (l1 ** 2 + l2 ** 2 - d_squared) / (2 * l1 * l2)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    gamma = np.arctan2(z_adj, r)
    theta2_sol1 = gamma - alpha
    theta3_sol1 = np.pi - beta
    theta2_sol2 = gamma + alpha
    theta3_sol2 = beta - np.pi
    if z < -0.3:
        if theta2_sol1 > 0:
            return (theta1, theta2_sol1, theta3_sol1)
        elif theta2_sol2 > 0:
            return (theta1, theta2_sol2, theta3_sol2)
    if z > 0.5 and r < 0.3:
        if theta2_sol1 > 0 and theta3_sol1 < 0:
            return (theta1, theta2_sol1, theta3_sol1)
        elif theta2_sol2 > 0 and theta3_sol2 < 0:
            return (theta1, theta2_sol2, theta3_sol2)
    x1 = l1 * np.sin(theta2_sol1) + l2 * np.sin(theta2_sol1 + theta3_sol1)
    z1 = l1 * np.cos(theta2_sol1) + l2 * np.cos(theta2_sol1 + theta3_sol1)
    x2 = l1 * np.sin(theta2_sol2) + l2 * np.sin(theta2_sol2 + theta3_sol2)
    z2 = l1 * np.cos(theta2_sol2) + l2 * np.cos(theta2_sol2 + theta3_sol2)
    dist1 = np.sqrt((r - x1) ** 2 + (z_adj - z1) ** 2)
    dist2 = np.sqrt((r - x2) ** 2 + (z_adj - z2) ** 2)
    if dist1 <= dist2:
        return (theta1, theta2_sol1, theta3_sol1)
    else:
        return (theta1, theta2_sol2, theta3_sol2)