def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    joint3 = r[2]
    j3_x = x
    j3_y = y
    j3_z = z - l3
    j2_x = j3_x - l2 * np.sin(joint3)
    j2_y = j3_y - l2 * np.cos(joint3)
    j2_z = j3_z
    joint1 = np.arctan2(j2_x, j2_z)
    rotated_x = j2_x * np.cos(-joint1) - j2_z * np.sin(-joint1)
    rotated_z = j2_x * np.sin(-joint1) + j2_z * np.cos(-joint1)
    distance = np.sqrt(rotated_x ** 2 + (rotated_z - l1) ** 2)
    joint2 = np.arctan2(rotated_x, rotated_z - l1)
    return (joint1, joint2, joint3)