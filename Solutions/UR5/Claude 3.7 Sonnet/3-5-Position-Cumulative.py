def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    joint1 = np.arctan2(x, z)
    r = np.sqrt(x ** 2 + z ** 2)
    l1_y = 0.093
    l2_z = 0.09465
    l3_y = 0.0823
    target_y = y - l1_y
    target_r = r
    d_squared = target_y ** 2 + target_r ** 2
    max_reach = l2_z + l3_y
    cos_joint3 = (l2_z ** 2 + l3_y ** 2 - d_squared) / (2 * l2_z * l3_y)
    cos_joint3 = np.clip(cos_joint3, -1.0, 1.0)
    joint3 = np.pi / 2 - np.arccos(cos_joint3)
    phi = np.arctan2(target_y, target_r)
    cos_alpha = (l2_z ** 2 + d_squared - l3_y ** 2) / (2 * l2_z * np.sqrt(d_squared))
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    joint2 = phi + alpha
    return (joint1, joint2, joint3)