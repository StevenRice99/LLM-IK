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
    r = np.sqrt(x ** 2 + z ** 2)
    cos_alpha = (l1 ** 2 + r ** 2 - l2 ** 2) / (2 * l1 * r)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = np.arccos(cos_alpha)
    cos_beta = (l1 ** 2 + l2 ** 2 - r ** 2) / (2 * l1 * l2)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    beta = np.arccos(cos_beta)
    gamma = np.arctan2(x, z)
    joint1_sol1 = gamma - alpha
    joint2_sol1 = np.pi - beta
    joint1_sol2 = gamma + alpha
    joint2_sol2 = beta - np.pi
    x1 = l1 * np.sin(joint1_sol1) + l2 * np.sin(joint1_sol1 + joint2_sol1)
    z1 = l1 * np.cos(joint1_sol1) + l2 * np.cos(joint1_sol1 + joint2_sol1)
    x2 = l1 * np.sin(joint1_sol2) + l2 * np.sin(joint1_sol2 + joint2_sol2)
    z2 = l1 * np.cos(joint1_sol2) + l2 * np.cos(joint1_sol2 + joint2_sol2)
    dist1 = np.sqrt((x - x1) ** 2 + (z - z1) ** 2)
    dist2 = np.sqrt((x - x2) ** 2 + (z - z2) ** 2)
    if dist1 <= dist2:
        return (joint1_sol1, joint2_sol1)
    else:
        return (joint1_sol2, joint2_sol2)