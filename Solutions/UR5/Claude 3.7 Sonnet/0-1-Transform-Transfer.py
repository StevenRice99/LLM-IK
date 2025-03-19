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
    rx, ry, rz = r
    l1 = 0.13585
    l2_y = -0.1197
    l2_z = 0.425
    A = l2_z
    B = l2_y
    if z < 0 and abs(rx) > 3.0:
        if ry > 0:
            theta2 = math.pi - ry
        else:
            theta2 = -math.pi - ry
        if rz > 0:
            theta1 = rz - math.pi
        else:
            theta1 = rz + math.pi
    else:
        theta2 = ry
        theta1 = rz
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)