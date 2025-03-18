import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    A = 0.425
    B = 0.39225
    x, _, z = p
    _, ry, __ = r
    C = A - B * math.cos(ry)
    D = B * math.sin(ry)
    E = A + B * math.cos(ry)
    F = B * math.sin(ry)
    det = A ** 2 - B ** 2
    sin_theta1 = (E * x - D * z) / det
    cos_theta1 = (-F * x + C * z) / det
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta2 = ry - theta1
    return (theta1, theta2)