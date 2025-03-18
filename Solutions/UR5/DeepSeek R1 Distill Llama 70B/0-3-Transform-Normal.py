def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    a = [0, 0.13585, -0.1197, 0]
    alpha = [0, 3 * 3 ** 0.5 / 2, 3.14159, 3.14159]
    d = [0, 0, 0.425, 0.39225]
    z = p[2]
    pitch = r[1]
    q3 = 0
    q4 = pitch - q3
    x = p[0]
    y = p[1]
    q2 = 0
    q1 = 0
    return (q1, q2, q3, q4)