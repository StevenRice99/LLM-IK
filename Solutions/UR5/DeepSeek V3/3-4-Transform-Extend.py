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
    L1 = 0.093
    L2 = 0.09465
    D = math.sqrt(px ** 2 + pz ** 2)
    cos_joint2 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    joint2 = math.atan2(math.sqrt(1 - cos_joint2 ** 2), cos_joint2)
    alpha = math.atan2(pz, px)
    beta = math.atan2(L2 * math.sin(joint2), L1 + L2 * math.cos(joint2))
    joint1 = alpha - beta
    return (joint1, joint2)