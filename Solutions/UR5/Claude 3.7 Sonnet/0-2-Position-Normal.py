def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    z_adj = z - 0.13585
    L2 = 0.425
    L3 = 0.39225
    D = np.sqrt(r ** 2 + z_adj ** 2)
    cos_theta3 = (r ** 2 + z_adj ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = -np.arccos(cos_theta3)
    beta = np.arctan2(z_adj, r)
    gamma = np.arctan2(L3 * np.sin(-theta3), L2 + L3 * np.cos(-theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)