import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.09465
    L2 = 0.0823
    d = math.sqrt(x ** 2 + y ** 2)
    if d > L1 + L2 or d < abs(L1 - L2):
        raise ValueError('Target position is not reachable.')
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta2 = math.acos(cos_theta2)
    alpha = math.atan2(y, x)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    return (theta1, theta2)