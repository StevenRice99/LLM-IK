def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    y_offset1 = -0.1197
    y_offset_tcp = 0.093
    y_target = y - y_offset_tcp
    joint1 = np.arctan2(x, z)
    r = np.sqrt(x ** 2 + z ** 2)
    y_diff = y_target - y_offset1
    d = np.sqrt(r ** 2 + y_diff ** 2)
    cos_joint3 = (r ** 2 + y_diff ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_joint3 = max(min(cos_joint3, 1.0), -1.0)
    joint3 = np.arccos(cos_joint3)
    beta = np.arccos((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d))
    alpha = np.arctan2(y_diff, r)
    joint2 = alpha - beta
    if z < 0 and x == 0:
        joint2 = np.pi - joint2
        joint3 = -joint3
    return (joint1, joint2, joint3)