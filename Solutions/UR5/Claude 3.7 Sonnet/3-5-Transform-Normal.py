def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    import math
    x, y, z = p
    l1 = 0.093
    l2 = 0.09465
    l3 = 0.0823
    if x == 0 and z == 0:
        theta1 = 0
    else:
        theta1 = math.atan2(x, z)
    dist_yz = math.sqrt(x ** 2 + z ** 2)
    y_rotated = y
    z_rotated = dist_yz
    theta3 = r[0]
    j3_y = y_rotated - l3 * math.sin(theta3)
    j3_z = z_rotated - l3 * math.cos(theta3)
    dist_j2_to_j3 = math.sqrt((j3_y - l1) ** 2 + j3_z ** 2)
    if dist_j2_to_j3 > l2:
        dist_j2_to_j3 = l2
    theta2 = math.atan2(j3_y - l1, j3_z) - math.asin(0)
    theta2 = r[2] - theta3 - math.pi / 2
    return (theta1, theta2, theta3)