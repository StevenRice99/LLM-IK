import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    tcp_offset = 0.093
    L1 = 0.425
    L2 = 0.39225
    x_adj = x
    y_adj = y - tcp_offset
    z_adj = z
    wrist_flip = abs(abs(rx) - math.pi) < 0.1 and abs(abs(rz) - math.pi) < 0.1
    d = math.sqrt(x_adj ** 2 + z_adj ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    alpha = math.atan2(x_adj, z_adj)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    if wrist_flip:
        theta2 = -theta2
        theta1 = alpha - beta
        if theta1 > 0:
            theta1 -= math.pi
        else:
            theta1 += math.pi
    else:
        theta1 = alpha - beta
    if wrist_flip:
        theta3 = ry - theta1 - theta2 + math.pi
    else:
        theta3 = ry - theta1 - theta2
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    theta3 = (theta3 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2, theta3)