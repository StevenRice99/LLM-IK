import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    theta3 = 0.0
    theta1 = math.atan2(x, z)
    r_xz = math.sqrt(x ** 2 + z ** 2)
    L_total = math.sqrt(L2 ** 2 + L3 ** 2)
    alpha = math.atan2(L3, L2)
    d = math.sqrt(r_xz ** 2 + (y - L2) ** 2)
    cos_beta = (L1 ** 2 + d ** 2 - L_total ** 2) / (2 * L1 * d)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    beta = math.acos(cos_beta)
    gamma = math.atan2(y - L2, r_xz)
    if y >= L2:
        theta2 = gamma - beta
    else:
        theta2 = gamma + beta
    if theta2 > math.pi:
        theta2 = theta2 - 2 * math.pi
    elif theta2 < -math.pi:
        theta2 = theta2 + 2 * math.pi
    return (theta1, theta2, theta3)