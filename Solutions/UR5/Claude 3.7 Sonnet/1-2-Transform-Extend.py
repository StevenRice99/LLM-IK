import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.425
    l2 = 0.39225
    d_squared = x ** 2 + z ** 2
    cos_theta2 = (d_squared - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    if abs(rx - math.pi) < 0.01 and abs(rz - math.pi) < 0.01:
        k1_pos = l1 + l2 * math.cos(theta2)
        k2_pos = l2 * math.sin(theta2)
        theta1_pos = math.atan2(x, z) - math.atan2(k2_pos, k1_pos)
        k1_neg = l1 + l2 * math.cos(-theta2)
        k2_neg = l2 * math.sin(-theta2)
        theta1_neg = math.atan2(x, z) - math.atan2(k2_neg, k1_neg)
        orient_pos = theta1_pos + theta2
        orient_neg = theta1_neg - theta2
        target_orient = ry + math.pi
        diff_pos = abs((orient_pos - target_orient + math.pi) % (2 * math.pi) - math.pi)
        diff_neg = abs((orient_neg - target_orient + math.pi) % (2 * math.pi) - math.pi)
        if diff_pos <= diff_neg:
            theta1 = theta1_pos
            theta2 = theta2
        else:
            theta1 = theta1_neg
            theta2 = -theta2
    else:
        k1_pos = l1 + l2 * math.cos(theta2)
        k2_pos = l2 * math.sin(theta2)
        theta1_pos = math.atan2(x, z) - math.atan2(k2_pos, k1_pos)
        k1_neg = l1 + l2 * math.cos(-theta2)
        k2_neg = l2 * math.sin(-theta2)
        theta1_neg = math.atan2(x, z) - math.atan2(k2_neg, k1_neg)
        orient_pos = theta1_pos + theta2
        orient_neg = theta1_neg - theta2
        diff_pos = abs((orient_pos - ry + math.pi) % (2 * math.pi) - math.pi)
        diff_neg = abs((orient_neg - ry + math.pi) % (2 * math.pi) - math.pi)
        if diff_pos <= diff_neg:
            theta1 = theta1_pos
            theta2 = theta2
        else:
            theta1 = theta1_neg
            theta2 = -theta2
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)