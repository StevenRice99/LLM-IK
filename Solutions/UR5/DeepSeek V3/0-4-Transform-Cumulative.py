import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1 = math.atan2(-x, y)
    theta2 = math.atan2(x, z)
    theta3 = math.atan2(x, z)
    theta4 = math.atan2(x, z)
    theta5 = math.atan2(y, x)
    return (theta1, theta2, theta3, theta4, theta5)