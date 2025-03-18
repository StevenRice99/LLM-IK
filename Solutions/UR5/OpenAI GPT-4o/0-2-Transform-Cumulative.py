import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    x2 = x
    y2 = y - 0.13585
    z2 = z
    theta2 = math.atan2(x2, z2)
    x3 = x2
    y3 = y2 + 0.1197
    z3 = z2 - 0.425
    theta3 = math.atan2(x3, z3)
    return (theta1, theta2, theta3)