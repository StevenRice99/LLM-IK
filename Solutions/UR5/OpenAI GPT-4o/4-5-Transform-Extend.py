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
    theta1 = math.atan2(y, x)
    l_tcp = 0.0823
    l_r2 = 0.09465
    y_eff = math.sqrt(x ** 2 + y ** 2) - l_tcp
    z_eff = z - l_r2
    theta2 = math.atan2(z_eff, y_eff)
    theta2 += r_y
    return (theta1, theta2)