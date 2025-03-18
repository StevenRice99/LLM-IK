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
    L1 = 0.39225
    L2 = 0.093
    x_adj = x - L2 * math.cos(ry)
    z_adj = z - L2 * math.sin(ry)
    theta1 = math.atan2(z_adj, x_adj)
    theta2 = ry - theta1
    return (theta1, theta2)