import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    theta1 = math.atan2(p[0], p[2])
    p2 = [p[0] - 0, p[1] - -0.1197, p[2] - 0.425]
    theta2 = math.atan2(p2[0], p2[2])
    p3 = [p[0] - 0, p[1] - 0, p[2] - 0.39225]
    theta3 = math.atan2(p3[0], p3[2])
    theta4 = math.atan2(r[1], r[0])
    return (theta1, theta2, theta3, theta4)