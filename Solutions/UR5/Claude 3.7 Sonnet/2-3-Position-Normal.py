def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.39225
    y_offset = 0.093
    y_target = y - y_offset
    joint1 = np.arctan2(x, z)
    r = np.sqrt(x ** 2 + z ** 2)
    joint2 = np.arcsin(r / l1) if r <= l1 else 0
    if z < 0 and x == 0:
        joint2 = np.pi - joint2
    return (joint1, joint2)