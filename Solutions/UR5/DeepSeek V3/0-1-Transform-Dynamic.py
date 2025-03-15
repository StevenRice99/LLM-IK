import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    length_link1 = 0.13585
    length_link2 = 0.425
    joint1 = math.atan2(y, x)
    d = math.sqrt(x ** 2 + y ** 2)
    D = math.sqrt(d ** 2 + (z - length_link1) ** 2)
    cos_theta2 = (D ** 2 - length_link1 ** 2 - length_link2 ** 2) / (2 * length_link1 * length_link2)
    cos_theta2 = max(-1.0, min(cos_theta2, 1.0))
    theta2 = math.acos(cos_theta2)
    joint2 = math.atan2(z - length_link1, d) - math.atan2(length_link2 * math.sin(theta2), length_link1 + length_link2 * math.cos(theta2))
    return (joint1, joint2)