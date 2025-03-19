def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    d1 = 0
    d2 = 0.13585
    d3 = -0.1197
    a3 = 0.425
    a4 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    d7 = 0.0823
    r_target = np.sqrt(x ** 2 + y ** 2)
    theta1 = np.arctan2(y, x)
    wc_x = x - d7 * np.sin(theta1)
    wc_y = y + d7 * np.cos(theta1)
    wc_z = z
    r_wc = np.sqrt(wc_x ** 2 + wc_y ** 2)
    wc_height = wc_z - d2
    planar_dist = np.sqrt(r_wc ** 2 + (wc_height - d3) ** 2)
    cos_theta3 = (planar_dist ** 2 - a3 ** 2 - a4 ** 2) / (2 * a3 * a4)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    alpha = np.arctan2(wc_height - d3, r_wc)
    beta = np.arccos((a3 ** 2 + planar_dist ** 2 - a4 ** 2) / (2 * a3 * planar_dist))
    theta2 = alpha - beta
    theta4 = -(theta2 + theta3)
    theta5 = -theta1
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)