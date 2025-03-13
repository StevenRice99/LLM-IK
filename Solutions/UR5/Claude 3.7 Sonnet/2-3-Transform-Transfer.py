def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    roll, pitch, yaw = r
    joint1 = np.arctan2(x, z)
    orientation_flipped = np.isclose(abs(roll), np.pi) and np.isclose(abs(yaw), np.pi)
    if orientation_flipped:
        joint2 = -pitch - joint1
        joint2 += np.pi
    else:
        joint2 = pitch - joint1
    joint1 = (joint1 + np.pi) % (2 * np.pi) - np.pi
    joint2 = (joint2 + np.pi) % (2 * np.pi) - np.pi
    return (joint1, joint2)