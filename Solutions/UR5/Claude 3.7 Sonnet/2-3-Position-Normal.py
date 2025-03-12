def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    L1 = 0.39225
    y_offset = 0.093
    y_adjusted = y - y_offset
    r = np.sqrt(x ** 2 + z ** 2)
    theta1 = np.arctan2(x, z)
    theta2 = 0.0
    return (theta1, theta2)