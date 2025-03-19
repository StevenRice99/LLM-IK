def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x_target, y_target, z_target = p
    l1 = 0.425
    l2 = 0.39225
    l3 = 0.093
    l4 = 0.09465
    l5 = 0.0823
    y_offset = -0.1197
    theta1 = np.arctan2(x_target, z_target)
    r_xz = np.sqrt(x_target ** 2 + z_target ** 2)
    theta5 = 0.0
    r_xy = np.sqrt(x_target ** 2 + y_target ** 2)
    max_reach = l1 + l2 + l3 + l4 + l5
    theta4 = np.arctan2(y_target, x_target) - theta1
    x3 = x_target - l5 * np.sin(theta1 + theta4)
    y3 = y_target - l5 * np.cos(theta1 + theta4) - l3
    z3 = z_target - l4
    d13 = np.sqrt(x3 ** 2 + (y3 - y_offset) ** 2 + z3 ** 2)
    cos_theta3 = (d13 ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    r3_xz = np.sqrt(x3 ** 2 + z3 ** 2)
    beta = np.arctan2(y3 - y_offset, r3_xz)
    gamma = np.arccos((l1 ** 2 + d13 ** 2 - l2 ** 2) / (2 * l1 * d13))
    theta2 = beta + gamma
    return (theta1, theta2, theta3, theta4, theta5)