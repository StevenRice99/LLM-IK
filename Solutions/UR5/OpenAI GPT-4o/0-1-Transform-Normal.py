import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    d1 = 0.425
    l2 = 0.13585
    l3 = 0.1197
    a = z - d1
    b = l2
    c = l3
    R = math.sqrt(b ** 2 + c ** 2)
    phi = math.atan2(c, b)
    theta_2 = math.asin(a / R) - phi
    theta_1 = math.atan2(y, x)
    return (theta_1, theta_2)