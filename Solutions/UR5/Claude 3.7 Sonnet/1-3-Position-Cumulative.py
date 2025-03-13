def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    y_offset_joint2 = -0.1197
    l1 = 0.425
    l2 = 0.39225
    y_offset_tcp = 0.093
    y_adjusted = y - y_offset_tcp
    r = np.sqrt(x ** 2 + z ** 2)
    phi = np.arctan2(x, z)
    y_target = y_adjusted - y_offset_joint2
    d = np.sqrt(r ** 2 + y_target ** 2)
    cos_theta2 = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    if z > 0 and x > 0:
        theta2 = np.arccos(cos_theta2)
    else:
        theta2 = -np.arccos(cos_theta2)
    alpha = np.arctan2(y_target, r)
    beta = np.arccos((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d))
    if theta2 > 0:
        theta1 = phi - alpha + beta
    else:
        theta1 = phi - alpha - beta
    theta3 = 0.0
    return (theta1, theta2, theta3)