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
    joint2 = ry
    if abs(abs(rx) - math.pi) < 0.1:
        joint1 = rz
        if z < 0:
            if joint2 > 0:
                joint2 = math.pi - joint2
            else:
                joint2 = -math.pi - joint2
    else:
        joint1 = math.atan2(x, y)
        if abs(joint1 - rz) > math.pi:
            if joint1 > rz:
                joint1 -= 2 * math.pi
            else:
                joint1 += 2 * math.pi
    joint1 = (joint1 + math.pi) % (2 * math.pi) - math.pi
    joint2 = (joint2 + math.pi) % (2 * math.pi) - math.pi
    return (joint1, joint2)