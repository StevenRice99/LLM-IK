import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.09465
    r = math.sqrt(x ** 2 + y ** 2)
    theta1 = math.atan2(y, x)
    x2 = r - L1
    z2 = z
    d = math.sqrt(x2 ** 2 + z2 ** 2)
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z2, x2)
    beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2 = alpha - beta
    theta4 = 0.0
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)