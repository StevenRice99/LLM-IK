import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    joint1 = math.atan2(y, x)
    x_joint2 = 0.13585 * math.cos(joint1)
    y_joint2 = 0.13585 * math.sin(joint1)
    z_joint2 = 0
    x_tcp = -0.1197 * math.sin(joint1) + 0.425 * math.cos(joint1)
    y_tcp = -0.1197 * math.cos(joint1) - 0.425 * math.sin(joint1)
    z_tcp = 0
    x_total = x_joint2 + x_tcp
    y_total = y_joint2 + y_tcp
    z_total = z_joint2 + z_tcp
    d = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    L = 0.441
    joint2 = math.acos((0.13585 ** 2 + d ** 2 - L ** 2) / (2 * 0.13585 * d))
    return (joint1, joint2)