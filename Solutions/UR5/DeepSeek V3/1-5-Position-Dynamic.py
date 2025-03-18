import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1, theta2, theta3 = inverse_kinematics_first_three((x, y, z))
    theta4, theta5 = inverse_kinematics_last_two((x, y, z), theta1, theta2, theta3)
    return (theta1, theta2, theta3, theta4, theta5)

def inverse_kinematics_first_three(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" for the first three joints.
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    L4 = 0.09465
    L5 = 0.0823
    x_adj = x
    y_adj = y - L5
    z_adj = z
    theta1 = math.atan2(y_adj, x_adj)
    d = math.sqrt(x_adj ** 2 + y_adj ** 2 + z_adj ** 2)
    cos_theta3 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_adj, math.sqrt(x_adj ** 2 + y_adj ** 2))
    beta = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)

def inverse_kinematics_last_two(p: tuple[float, float, float], theta1: float, theta2: float, theta3: float) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" for the last two joints.
    :param p: The position to reach in the form [x, y, z].
    :param theta1: The value of the first joint.
    :param theta2: The value of the second joint.
    :param theta3: The value of the third joint.
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    L4 = 0.09465
    L5 = 0.0823
    x3 = L1 * math.cos(theta1) * math.sin(theta2) + L2 * math.cos(theta1) * math.sin(theta2 + theta3)
    y3 = L1 * math.sin(theta1) * math.sin(theta2) + L2 * math.sin(theta1) * math.sin(theta2 + theta3)
    z3 = L1 * math.cos(theta2) + L2 * math.cos(theta2 + theta3)
    dx = x - x3
    dy = y - y3
    dz = z - z3
    theta4 = math.atan2(dy, dx)
    theta5 = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))
    return (theta4, theta5)