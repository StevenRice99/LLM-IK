import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    D = px ** 2 + pz ** 2 - 0.3345
    cos_q2 = D / 0.334
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2 = math.acos(cos_q2)
    q1 = ry - q2
    return (q1, q2)