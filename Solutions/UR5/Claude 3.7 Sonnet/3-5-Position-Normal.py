def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.093
    l2 = 0.09465
    l3 = 0.0823
    theta1 = np.arctan2(x, z)
    r_xz = np.sqrt(x ** 2 + z ** 2)
    y_rel = y - l1
    d = np.sqrt(r_xz ** 2 + y_rel ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    alpha = np.arctan2(y_rel, r_xz)
    beta = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)