import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    L4 = 0.09465
    L5 = 0.0823
    y_offset = 0.1197
    r_xz = math.sqrt(x ** 2 + z ** 2)
    theta1 = math.atan2(x, z)
    arm_length = math.sqrt(L2 ** 2 + L3 ** 2 + L4 ** 2 + L5 ** 2)
    d = math.sqrt(r_xz ** 2 + (y + y_offset) ** 2)
    cos_theta2 = (d ** 2 + L1 ** 2 - arm_length ** 2) / (2 * L1 * d)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    angle_at_joint2 = math.acos(cos_theta2)
    alpha = math.atan2(y + y_offset, r_xz)
    theta2 = alpha - angle_at_joint2
    cos_theta3 = (L1 ** 2 + arm_length ** 2 - d ** 2) / (2 * L1 * arm_length)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.pi - math.acos(cos_theta3)
    theta4 = -theta2 - theta3
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)