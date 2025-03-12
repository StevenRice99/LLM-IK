import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    z_offset = z - l3
    d = math.sqrt(r ** 2 + z_offset ** 2)
    alpha = math.atan2(z_offset, r)
    cos_beta = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_beta = max(-1, min(1, cos_beta))
    beta = math.acos(cos_beta)
    cos_gamma = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
    cos_gamma = max(-1, min(1, cos_gamma))
    gamma = math.acos(cos_gamma)
    theta2 = alpha + beta
    theta3 = math.pi - gamma
    theta3 = -(theta2 + theta3)
    return (theta1, theta2, theta3)