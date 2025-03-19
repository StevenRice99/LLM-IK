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
    theta1 = math.atan2(x, z)
    if abs(rx - math.pi) < 1e-10 and abs(rz - math.pi) < 1e-10:
        theta2 = -ry - theta1 + math.pi
        while theta2 > 2 * math.pi:
            theta2 -= 2 * math.pi
        while theta2 < -2 * math.pi:
            theta2 += 2 * math.pi
    else:
        theta2 = ry - theta1
    return (theta1, theta2)