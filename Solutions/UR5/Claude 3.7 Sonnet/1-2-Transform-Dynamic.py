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
    target_dist = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (target_dist ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_pos = math.acos(cos_theta2)
    theta2_neg = -math.acos(cos_theta2)
    phi = math.atan2(x, z)
    psi_pos = math.atan2(l2 * math.sin(theta2_pos), l1 + l2 * math.cos(theta2_pos))
    theta1_pos = phi - psi_pos
    psi_neg = math.atan2(l2 * math.sin(theta2_neg), l1 + l2 * math.cos(theta2_neg))
    theta1_neg = phi - psi_neg
    theta1 = theta1_pos
    theta2 = theta2_pos
    theta1_alt = theta1_neg
    theta2_alt = theta2_neg
    if abs(rx - math.pi) < 0.1:
        expected_ry1 = theta1 + theta2
        if abs(expected_ry1 - ry) > math.pi:
            if expected_ry1 > ry:
                expected_ry1 -= 2 * math.pi
            else:
                expected_ry1 += 2 * math.pi
        expected_ry2 = theta1_alt + theta2_alt
        if abs(expected_ry2 - ry) > math.pi:
            if expected_ry2 > ry:
                expected_ry2 -= 2 * math.pi
            else:
                expected_ry2 += 2 * math.pi
        if abs(expected_ry1 - ry) <= abs(expected_ry2 - ry):
            return (theta1, theta2)
        else:
            return (theta1_alt, theta2_alt)
    else:
        expected_ry1 = theta1 + theta2
        if abs(expected_ry1 - ry) > math.pi:
            if expected_ry1 > ry:
                expected_ry1 -= 2 * math.pi
            else:
                expected_ry1 += 2 * math.pi
        expected_ry2 = theta1_alt + theta2_alt
        if abs(expected_ry2 - ry) > math.pi:
            if expected_ry2 > ry:
                expected_ry2 -= 2 * math.pi
            else:
                expected_ry2 += 2 * math.pi
        if abs(expected_ry1 - ry) <= abs(expected_ry2 - ry):
            return (theta1, theta2)
        else:
            return (theta1_alt, theta2_alt)