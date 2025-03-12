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
    joint2_height = 0.13585
    target_r = r
    target_z = z - joint2_height
    l2 = 0.425
    l3 = 0.39225
    d_squared = target_r ** 2 + target_z ** 2
    cos_theta3 = (d_squared - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    beta = np.arctan2(target_z, target_r)
    gamma = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)