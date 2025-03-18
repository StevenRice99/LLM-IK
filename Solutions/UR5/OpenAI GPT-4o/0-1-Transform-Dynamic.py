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
    theta_1 = math.atan2(y, x)
    x_prime = x * math.cos(theta_1) + y * math.sin(theta_1)
    z_prime = z
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    l2 = 0.425
    l1 = 0.13585
    cos_theta_2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta_2 = math.acos(cos_theta_2)
    theta_2 = theta_2 - math.atan2(z_prime, x_prime)
    return (theta_1, theta_2)