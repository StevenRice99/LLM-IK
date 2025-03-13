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
    z_eff = z - l1
    d = math.sqrt(r ** 2 + z_eff ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    cos_beta = (l2 ** 2 + d ** 2 - l3 ** 2) / (2 * l2 * d)
    cos_beta = max(-1, min(1, cos_beta))
    beta = math.acos(cos_beta)
    alpha = math.atan2(z_eff, r)
    theta2 = alpha + beta
    return (theta1, theta2, theta3)