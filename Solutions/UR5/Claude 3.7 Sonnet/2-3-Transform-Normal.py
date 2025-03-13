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
    joint1 = math.atan2(x, z)
    flipped = abs(abs(rx) - math.pi) < 1e-06 and abs(abs(rz) - math.pi) < 1e-06
    if flipped:
        joint2_base = ry + math.pi - joint1
        while joint2_base > math.pi:
            joint2_base -= 2 * math.pi
        while joint2_base < -math.pi:
            joint2_base += 2 * math.pi
        if ry > 0:
            joint2 = joint2_base + 2 * math.pi
        else:
            joint2 = joint2_base - 2 * math.pi
    else:
        joint2 = ry - joint1
    return (joint1, joint2)