import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d = y - L1
    sin_theta2 = max(-1, min(1, d / L2))
    theta2 = math.asin(sin_theta2)
    cos_theta2 = math.sqrt(1 - sin_theta2 ** 2)
    r = math.sqrt(x ** 2 + z ** 2)
    if r != 0:
        theta1 = math.atan2(x, z)
    else:
        theta1 = 0
    return (theta1, theta2)