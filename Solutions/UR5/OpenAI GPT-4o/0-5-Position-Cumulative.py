import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.093
    L5 = 0.09465
    TCP_offset = 0.0823
    theta1 = math.atan2(-x, y)
    x_prime = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_prime, x_prime)
    beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2 = alpha - beta
    theta4 = 0
    theta5 = math.atan2(y, x)
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)