def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    rx, ry, rz = r
    is_flipped = abs(abs(rx) - math.pi) < 1e-06
    if is_flipped:
        if rz >= 0:
            theta1 = rz - math.pi
        else:
            theta1 = rz + math.pi
        if rx > 0:
            theta2 = math.pi - ry
        else:
            theta2 = -math.pi - ry
    else:
        theta1 = rz
        theta2 = ry
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)