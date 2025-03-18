import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    cos_q2 = (z - 0.09465) / 0.39225
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2 = math.acos(cos_q2)
    A = 0.39225 * math.sin(q2)
    B = 0.093
    numerator = y * A - x * B
    denominator = x * A + y * B
    q3 = math.atan2(numerator, denominator)
    q1 = 0.0
    return (q1, q2, q3)