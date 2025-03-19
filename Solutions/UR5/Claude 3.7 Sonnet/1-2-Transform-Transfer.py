def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.425
    L2 = 0.39225
    sin_ry = math.sin(ry)
    cos_ry = math.cos(ry)
    x_adj = x - L2 * sin_ry
    z_adj = z - L2 * cos_ry
    theta1 = math.atan2(x_adj, z_adj)
    theta2 = ry - theta1
    return (theta1, theta2)