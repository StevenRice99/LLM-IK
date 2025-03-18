import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    theta1 = math.atan2(px, pz)
    x_prime = math.cos(-theta1) * px - math.sin(-theta1) * pz
    z_prime = math.sin(-theta1) * px + math.cos(-theta1) * pz
    y_prime = py - 0.093
    z_prime -= 0.09465
    theta2 = rz
    return (theta1, theta2)