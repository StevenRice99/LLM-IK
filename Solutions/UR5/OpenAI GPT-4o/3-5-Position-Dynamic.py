import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    d = math.sqrt(y ** 2 + z ** 2)
    l1 = 0.093
    l2 = 0.09465
    cos_theta3 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(y, z)
    beta = math.asin(l2 * math.sin(theta3) / d)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)