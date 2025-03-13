def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.093
    l2 = 0.09465
    joint1 = math.atan2(x, z)
    joint2 = rz - joint1
    joint1 = (joint1 + np.pi) % (2 * np.pi) - np.pi
    joint2 = (joint2 + np.pi) % (2 * np.pi) - np.pi
    return (joint1, joint2)