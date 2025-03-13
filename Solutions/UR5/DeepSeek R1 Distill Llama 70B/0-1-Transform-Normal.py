import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    pz = p[2]
    theta2 = math.acos(pz / 0.425)
    A = 0.425 * math.sin(theta2)
    B = 0.1197
    px, py = (p[0], p[1])
    denominator = A ** 2 + B ** 2
    cos_theta1 = (A * px - B * py) / denominator
    sin_theta1 = (A * py + B * px) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)