def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
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
    theta3 = 0.0
    r = np.sqrt(x ** 2 + z ** 2)
    phi = np.arctan2(x, z)
    cos_alpha = (l1 ** 2 + l3 ** 2 - r ** 2) / (2 * l1 * l3)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    cos_beta = (l1 ** 2 + r ** 2 - l3 ** 2) / (2 * l1 * r)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    theta1 = phi - beta
    theta2 = np.pi - alpha
    return (theta1, theta2, theta3)