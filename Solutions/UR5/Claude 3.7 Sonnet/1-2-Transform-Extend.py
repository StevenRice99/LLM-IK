def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    x = p[0]
    y = p[1]
    z = p[2]
    rx, ry, rz = r
    l1 = 0.425
    l2 = 0.39225
    flipped = abs(abs(rx) - math.pi) < 1e-06 and abs(abs(rz) - math.pi) < 1e-06
    if flipped:
        theta2 = math.pi - ry
    else:
        theta2 = ry
    x_wrist = x - l2 * math.sin(theta2)
    z_wrist = z - l2 * math.cos(theta2)
    dist = math.sqrt(x_wrist ** 2 + z_wrist ** 2)
    if dist > l1:
        theta1 = math.atan2(x_wrist, z_wrist)
    else:
        theta1 = math.atan2(x_wrist, z_wrist)
        if flipped:
            theta1 = theta1 + math.pi
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)