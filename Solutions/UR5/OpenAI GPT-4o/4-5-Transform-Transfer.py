import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    L1 = 0.09465
    L2 = 0.0823
    theta1 = math.atan2(-x, y)
    theta2 = r_z - math.pi / 2
    return (theta1, theta2)