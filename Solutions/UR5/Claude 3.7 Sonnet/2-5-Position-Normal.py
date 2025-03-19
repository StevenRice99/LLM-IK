def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    l4 = 0.0823
    theta3 = np.arctan2(x, y)
    wc_x = x - l4 * np.sin(theta3)
    wc_y = y - l4 * np.cos(theta3)
    wc_z = z
    r_xy = np.sqrt(wc_x ** 2 + wc_y ** 2)
    h = wc_z - l1
    d = np.sqrt(h ** 2 + r_xy ** 2)
    theta1 = np.arctan2(h, r_xy)
    cos_theta2 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    theta2 = np.pi - np.arccos(np.clip(cos_theta2, -1.0, 1.0))
    theta4 = -theta1 - theta2
    return (theta1, theta2, theta3, theta4)