def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.13585
    l2z = 0.425
    l2y = -0.1197
    l3 = 0.39225
    l4y = 0.093
    l5z = 0.09465
    if x == 0 and y == 0:
        theta1 = 0
    else:
        theta1 = np.arctan2(y, x)
    wc_x = x
    wc_y = y
    wc_z = z - l5z
    r = np.sqrt(wc_x ** 2 + wc_y ** 2)
    dx = r
    dz = wc_z - l1
    D = np.sqrt(dx ** 2 + dz ** 2)
    a2 = np.sqrt(l2z ** 2 + l2y ** 2)
    a3 = np.sqrt(l3 ** 2 + l4y ** 2)
    cos_theta3 = (D ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    gamma = np.arctan2(dx, dz)
    beta = np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    theta2 = gamma - beta
    theta4 = -(theta2 + theta3)
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)