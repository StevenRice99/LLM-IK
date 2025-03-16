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
    r_target = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (x ** 2 + z ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    if abs(ry) < math.pi / 2 or abs(ry - math.pi) < math.pi / 2 or abs(ry + math.pi) < math.pi / 2:
        theta2 = -math.acos(cos_theta2)
    else:
        theta2 = math.acos(cos_theta2)
    phi = math.atan2(x, z)
    psi = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = phi - psi
    if abs(rx - math.pi) < math.pi / 2 and abs(rz - math.pi) < math.pi / 2 or (abs(rx + math.pi) < math.pi / 2 and abs(rz + math.pi) < math.pi / 2):
        theta1 = theta1 + math.pi
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)