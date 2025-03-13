import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.093
    L2 = 0.09465
    L3 = 0.0823
    theta1 = math.atan2(x, z)
    r = math.sqrt(x ** 2 + z ** 2)
    D = (r ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(D) > 1:
        D = 1 if D > 1 else -1
    theta2 = math.atan2(-math.sqrt(1 - D ** 2), D)
    theta3 = math.atan2(y, r) - math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    return (theta1, theta2, theta3)