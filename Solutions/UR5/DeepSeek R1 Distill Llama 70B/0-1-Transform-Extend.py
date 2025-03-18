import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    cos_theta2 = z / 0.425
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    sin_theta1 = (0.425 * sin_theta2 - x) / 0.13585
    cos_theta1 = (y + 0.1197) / 0.13585
    if not abs(sin_theta1 ** 2 + cos_theta1 ** 2 - 1) < 1e-06:
        raise ValueError('The target position is not reachable.')
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)