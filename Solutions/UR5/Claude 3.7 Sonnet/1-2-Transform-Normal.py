def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.425
    l2 = 0.39225
    r_squared = x ** 2 + z ** 2
    r = np.sqrt(r_squared)
    cos_theta2 = (r_squared - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    if abs(ry) < np.pi / 2 or abs(ry - np.pi) < np.pi / 2:
        theta2 = -np.arccos(cos_theta2)
    else:
        theta2 = np.arccos(cos_theta2)
    if abs(rx) < 0.1 and abs(rz) < 0.1:
        beta = np.arctan2(x, z)
        alpha = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = beta - alpha
    else:
        beta = np.arctan2(-x, -z)
        alpha = np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
        theta1 = beta + alpha
    if abs(rx - np.pi) < 0.1 and abs(rz - np.pi) < 0.1:
        if x < 0 and z < 0:
            theta1 = theta1 + np.pi
    return (theta1, theta2)