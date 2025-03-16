def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    l1 = 0.13585
    l2_x = 0
    l2_y = -0.1197
    l2_z = 0.425
    r_xz = math.sqrt(x ** 2 + z ** 2)
    l2_xz = math.sqrt(l2_x ** 2 + l2_z ** 2)
    theta2 = math.atan2(r_xz, y - l1 - l2_y)
    theta2 = theta2 - math.atan2(l2_z, l2_x)
    theta1 = math.atan2(x, z)
    return (theta1, theta2)