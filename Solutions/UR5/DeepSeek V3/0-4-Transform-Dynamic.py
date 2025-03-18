import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.093
    L5 = 0.09465
    theta1 = math.atan2(y, x)
    x2 = x - L1 * math.cos(theta1)
    y2 = y - L1 * math.sin(theta1)
    z2 = z
    d = math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
    arg = (L2 ** 2 + d ** 2 - L3 ** 2) / (2 * L2 * d)
    arg = max(min(arg, 1), -1)
    theta2 = math.acos(arg) + math.atan2(z2, math.sqrt(x2 ** 2 + y2 ** 2))
    arg = (L2 ** 2 + L3 ** 2 - d ** 2) / (2 * L2 * L3)
    arg = max(min(arg, 1), -1)
    theta3 = math.acos(arg)
    theta4 = ry - (theta2 + theta3)
    theta5 = rz
    return (theta1, theta2, theta3, theta4, theta5)