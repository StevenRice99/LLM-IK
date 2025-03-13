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
    r_xz = np.sqrt(x ** 2 + z ** 2)
    if np.isclose(r_xz, 0):
        theta1 = 0
    else:
        theta1 = np.arctan2(x, z)
    y_from_base = y - l1
    x_from_base = r_xz
    r = np.sqrt(x_from_base ** 2 + y_from_base ** 2)
    cos_theta3 = (r ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    beta = np.arctan2(y_from_base, x_from_base)
    gamma = np.arccos((l2 ** 2 + r ** 2 - l3 ** 2) / (2 * l2 * r))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)