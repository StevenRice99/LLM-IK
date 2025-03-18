import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    _, ry, _ = r
    l1 = 0.425
    l2 = 0.39225
    theta1 = math.atan2(x, z)
    x_prime = math.sqrt(x ** 2 + z ** 2)
    z_prime = y
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    theta2 = theta2 - ry
    return (theta1, theta2)