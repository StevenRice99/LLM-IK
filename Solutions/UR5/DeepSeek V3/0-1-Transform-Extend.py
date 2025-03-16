import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = math.atan2(y, x)
    x2 = 0.13585 * math.cos(theta1)
    y2 = 0.13585 * math.sin(theta1)
    z2 = 0
    dx = x - x2
    dy = y - y2
    dz = z - z2
    theta2 = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))
    theta2 += pitch
    return (theta1, theta2)