import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    theta1 = math.atan2(x, z)
    x3 = x - L3 * math.sin(theta1)
    z3 = z - L3 * math.cos(theta1)
    D = math.sqrt(x3 ** 2 + z3 ** 2)
    cos_theta2 = (L1 ** 2 + L2 ** 2 - D ** 2) / (2 * L1 * L2)
    cos_theta2 = max(-1.0, min(cos_theta2, 1.0))
    theta2 = math.acos(cos_theta2)
    theta3 = math.atan2(y, x) - theta1 - theta2
    return (theta1, theta2, theta3)