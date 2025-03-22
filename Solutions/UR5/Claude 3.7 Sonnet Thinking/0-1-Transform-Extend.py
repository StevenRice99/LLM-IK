import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    cos_theta2 = max(-1.0, min(1.0, z / 0.425))
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sin(theta2)
    a = 0.425 * sin_theta2
    b = 0.01615
    numerator = a * y - b * x
    denominator = a * x + b * y
    theta1 = math.atan2(numerator, denominator)
    return (theta1, theta2)