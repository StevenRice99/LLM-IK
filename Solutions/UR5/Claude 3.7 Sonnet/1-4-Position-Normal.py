def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    import math
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    l3 = 0.093
    l4 = 0.09465
    y_offset = -0.1197
    wrist_x = x
    wrist_y = y
    wrist_z = z - l4
    r_xz = np.sqrt(wrist_x ** 2 + wrist_z ** 2)
    basic_theta1 = np.arctan2(wrist_x, wrist_z)
    theta1 = np.arctan2(x, z)
    r = np.sqrt(wrist_x ** 2 + wrist_z ** 2)
    y_effective = wrist_y - y_offset
    d = np.sqrt(r ** 2 + y_effective ** 2)
    if d > l1 + l2:
        d = l1 + l2 - 0.0001
    cos_theta3 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    if z < 0:
        theta3 = np.arccos(cos_theta3)
    else:
        theta3 = -np.arccos(cos_theta3)
    cos_beta = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    alpha = np.arctan2(y_effective, r)
    if z < 0:
        theta2 = alpha + beta
    else:
        theta2 = alpha - beta
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)