def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    x, y, z = p
    joint1 = math.atan2(x, z)
    distance_xz = math.sqrt(x ** 2 + z ** 2)
    link2_length = 0.39225
    _, pitch, _ = r
    joint2 = pitch - joint1
    joint1 = (joint1 + math.pi) % (2 * math.pi) - math.pi
    joint2 = (joint2 + math.pi) % (2 * math.pi) - math.pi
    return (joint1, joint2)