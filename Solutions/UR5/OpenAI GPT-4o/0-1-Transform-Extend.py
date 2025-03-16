import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    _, pitch, _ = r
    l1 = 0.13585
    l2 = 0.425
    theta1 = math.atan2(-x, y)
    x_eff = math.sqrt(x ** 2 + y ** 2)
    z_eff = z - l1
    d = math.sqrt(x_eff ** 2 + z_eff ** 2)
    cos_angle = max(-1, min(1, l2 / d))
    theta2 = math.atan2(z_eff, x_eff) - math.acos(cos_angle)
    theta2 += pitch
    return (theta1, theta2)