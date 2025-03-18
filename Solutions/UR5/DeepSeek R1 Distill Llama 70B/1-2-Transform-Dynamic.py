import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    A = (x - 0.39225 * math.sin(ry)) / 0.425
    B = (z - 0.39225 * math.cos(ry)) / 0.425
    theta1 = math.atan2(A, B)
    theta2 = ry - theta1
    return (theta1, theta2)