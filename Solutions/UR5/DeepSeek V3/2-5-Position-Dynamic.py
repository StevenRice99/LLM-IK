import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    s = z - L1
    D_numerator = r ** 2 + s ** 2 - L2 ** 2 - L3 ** 2
    D_denominator = 2 * L2 * L3
    D = D_numerator / D_denominator
    if abs(D) > 1:
        D = -D
        if abs(D) > 1:
            raise ValueError('Target position is not reachable.')
    theta3 = math.atan2(math.sqrt(1 - D ** 2), D)
    theta2 = math.atan2(s, r) - math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)