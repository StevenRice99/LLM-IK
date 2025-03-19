def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    joint3 = 0.0
    r = np.sqrt(x ** 2 + (z - l1) ** 2)
    cos_joint2 = (x ** 2 + (z - l1) ** 2 - l3 ** 2) / (2 * l1 * l3)
    cos_joint2 = np.clip(cos_joint2, -1.0, 1.0)
    joint2 = np.arccos(cos_joint2)
    phi = np.arctan2(x, z - l1)
    psi = np.arctan2(l3 * np.sin(joint2), l1 + l3 * np.cos(joint2))
    joint1 = phi - psi
    joint1 = (joint1 + np.pi) % (2 * np.pi) - np.pi
    joint2 = (joint2 + np.pi) % (2 * np.pi) - np.pi
    return (joint1, joint2, joint3)