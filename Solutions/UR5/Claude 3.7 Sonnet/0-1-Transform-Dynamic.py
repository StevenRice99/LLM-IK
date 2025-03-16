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
    quadrant = 0
    if x >= 0 and y >= 0:
        quadrant = 1
    elif x < 0 and y >= 0:
        quadrant = 2
    elif x < 0 and y < 0:
        quadrant = 3
    else:
        quadrant = 4
    if x >= 0:
        theta1 = rz - math.pi
    else:
        theta1 = rz + math.pi
    while theta1 > math.pi:
        theta1 -= 2 * math.pi
    while theta1 < -math.pi:
        theta1 += 2 * math.pi
    if z < 0:
        if ry > 0:
            theta2 = math.pi - ry
        else:
            theta2 = -math.pi - ry
    else:
        theta2 = ry
    while theta2 > math.pi:
        theta2 -= 2 * math.pi
    while theta2 < -math.pi:
        theta2 += 2 * math.pi
    return (theta1, theta2)