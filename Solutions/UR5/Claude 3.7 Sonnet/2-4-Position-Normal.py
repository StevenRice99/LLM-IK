def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    theta3 = 0.0
    r = np.sqrt(x ** 2 + z ** 2)
    l1 = 0.39225
    l3 = 0.09465
    cos_theta2 = (r ** 2 - l1 ** 2 - l3 ** 2) / (2 * l1 * l3)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    phi = np.arctan2(l3 * np.sin(theta2), l1 + l3 * np.cos(theta2))
    theta1 = np.arctan2(x, z) - phi
    return (theta1, theta2, theta3)