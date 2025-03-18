import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    theta1 = math.atan2(px, pz)
    distance_xz = math.sqrt(px ** 2 + pz ** 2)
    pyz = (py, distance_xz)
    l2 = 0.093
    l3 = 0.17695
    d = math.sqrt(pyz[0] ** 2 + pyz[1] ** 2)
    if not abs(l2 - l3) <= d <= l2 + l3:
        raise ValueError('Target position is unreachable')
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(pyz[0], pyz[1])
    beta = math.atan2(l3 * math.sin(theta3), l2 + l3 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)