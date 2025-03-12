def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    Y_OFFSET1 = -0.1197
    Y_OFFSET_TCP = 0.093
    y_adjusted = y - Y_OFFSET_TCP
    theta1 = np.arctan2(x, z)
    r = np.sqrt(x ** 2 + z ** 2)
    r2 = r
    y2 = y_adjusted - Y_OFFSET1
    cos_theta3 = (r2 ** 2 + y2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    beta = np.arctan2(y2, r2)
    gamma = np.arctan2(L2 * np.sin(theta3), L1 + L2 * np.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)