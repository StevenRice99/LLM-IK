import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    q1 = math.atan2(p[0], p[2])
    x = p[0] - 0.093 * math.cos(q1)
    z = p[2] + 0.093 * math.sin(q1)
    q2 = math.atan2(x, z)
    x = x - 0.425 * math.cos(q2)
    z = z + 0.425 * math.sin(q2)
    q3 = math.atan2(x, z)
    q4 = math.atan2(r[1], r[0])
    return (q1, q2, q3, q4)