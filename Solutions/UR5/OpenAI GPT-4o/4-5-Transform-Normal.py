import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.09465
    L2 = 0.0823
    theta_1 = math.atan2(y, x)
    cos_theta_2 = (z - L1) / L2
    theta_2 = math.acos(cos_theta_2)
    theta_1 += rz
    theta_2 += rx
    return (theta_1, theta_2)