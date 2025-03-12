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
    if abs(r_xz) < 1e-10:
        theta1 = 0
    else:
        theta1 = np.arctan2(x, z)
    y_from_j2 = y - l1
    xz_from_j2 = r_xz
    r = np.sqrt(y_from_j2 ** 2 + xz_from_j2 ** 2)
    max_reach = l2 + l3
    cos_theta3 = (r ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    gamma = np.arctan2(y_from_j2, xz_from_j2)
    sin_alpha = l3 * np.sin(theta3) / r
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    alpha = np.arcsin(sin_alpha)
    theta2 = gamma - alpha
    theta3 = np.pi - theta3
    return (theta1, theta2, theta3)