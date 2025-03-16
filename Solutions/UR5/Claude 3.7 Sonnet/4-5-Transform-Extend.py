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
    joint2 = rx
    if abs(math.sin(joint2)) < 0.5:
        joint1 = -rz + math.pi / 2
    else:
        joint1 = -rz - math.pi / 2
    while joint1 > math.pi:
        joint1 -= 2 * math.pi
    while joint1 < -math.pi:
        joint1 += 2 * math.pi
    return (joint1, joint2)