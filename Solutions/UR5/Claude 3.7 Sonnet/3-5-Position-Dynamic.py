import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    theta1 = math.atan2(px, pz)
    r_xz = math.sqrt(px ** 2 + pz ** 2)
    j3y = py - 0.0823 * math.cos(theta1)
    r_target = math.sqrt((r_xz - 0.0823 * math.sin(theta1)) ** 2 + (py - 0.093 - 0.0823 * math.cos(theta1)) ** 2)
    l2 = 0.09465
    l3 = 0.0823
    cos_theta3 = (l2 ** 2 + l3 ** 2 - r_target ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.pi - math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    angle_to_target = math.atan2(py - 0.093, r_xz)
    sin_beta = l3 * sin_theta3 / r_target
    sin_beta = max(min(sin_beta, 1.0), -1.0)
    beta = math.asin(sin_beta)
    theta2 = angle_to_target - beta
    return (theta1, theta2, theta3)