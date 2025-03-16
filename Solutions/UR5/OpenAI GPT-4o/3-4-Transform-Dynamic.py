import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(z, x)
    x_prime = math.sqrt(x ** 2 + z ** 2)
    z_prime = 0
    sin_theta2 = (y - 0.093) / 0.09465
    theta2 = math.asin(sin_theta2)
    return (theta1, theta2)