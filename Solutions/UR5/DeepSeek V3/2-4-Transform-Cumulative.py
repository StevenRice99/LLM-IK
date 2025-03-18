import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
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
    L3 = 0.09465
    theta1 = math.atan2(x, z)
    x_prime = x - L2 * math.sin(theta1)
    z_prime = z - L2 * math.cos(theta1)
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta2 = (L1 ** 2 + d ** 2 - L3 ** 2) / (2 * L1 * d)
    if cos_theta2 < -1:
        cos_theta2 = -1
    elif cos_theta2 > 1:
        cos_theta2 = 1
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta3 = rz
    return (theta1, theta2, theta3)