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
    distance = math.sqrt(x ** 2 + y ** 2)
    if not abs(L1 - L2) <= distance <= L1 + L2:
        raise ValueError('Target position is not reachable.')
    theta1 = math.atan2(y, x)
    D = (x ** 2 + y ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta2 = math.atan2(math.sqrt(1 - D ** 2), D)
    theta3 = math.atan2(y - L1 * math.sin(theta1) - L2 * math.sin(theta1 + theta2), x - L1 * math.cos(theta1) - L2 * math.cos(theta1 + theta2)) - theta1 - theta2
    return (theta1, theta2, theta3)