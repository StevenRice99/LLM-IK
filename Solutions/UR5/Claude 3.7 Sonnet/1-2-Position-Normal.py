def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    y_offset = -0.1197
    l1 = 0.425
    l2 = 0.39225
    y_target = y - y_offset
    r = np.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    phi = np.arctan2(x, z)
    psi = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    theta1 = phi - psi
    return (theta1, theta2)