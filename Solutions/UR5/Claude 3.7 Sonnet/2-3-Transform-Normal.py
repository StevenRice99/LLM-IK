def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    joint1 = np.arctan2(x, z)
    is_flipped = abs(rx - np.pi) < 1e-06 and abs(rz - np.pi) < 1e-06
    if is_flipped:
        joint2 = ry - joint1 + np.pi
        while joint2 > np.pi:
            joint2 -= 2 * np.pi
        while joint2 < -np.pi:
            joint2 += 2 * np.pi
    else:
        joint2 = ry - joint1
        while joint2 > np.pi:
            joint2 -= 2 * np.pi
        while joint2 < -np.pi:
            joint2 += 2 * np.pi
    while joint1 > np.pi:
        joint1 -= 2 * np.pi
    while joint1 < -np.pi:
        joint1 += 2 * np.pi
    return (joint1, joint2)