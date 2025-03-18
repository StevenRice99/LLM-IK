import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    z_tcp = 0.425
    phi = math.acos(z / z_tcp)
    sin_phi = math.sqrt(1 - (z / z_tcp) ** 2)
    A = 0.425 * sin_phi
    B = 0.1197
    denominator = A ** 2 + B ** 2
    cos_theta = (A * x - B * y) / denominator
    sin_theta = (A * y + B * x) / denominator
    theta = math.atan2(sin_theta, cos_theta)
    return (theta, phi)