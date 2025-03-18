import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    theta1 = math.atan2(-p[0], p[1])
    x = p[0] * math.cos(theta1) + p[1] * math.sin(theta1)
    z = p[2]
    theta2 = math.atan2(x, z)
    x = x * math.cos(theta2) - z * math.sin(theta2)
    z = x * math.sin(theta2) + z * math.cos(theta2)
    theta3 = math.atan2(x, z)
    theta4 = math.atan2(p[0], p[1])
    return (theta1, theta2, theta3, theta4)