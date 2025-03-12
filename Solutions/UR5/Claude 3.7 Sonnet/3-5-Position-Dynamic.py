import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.093
    l2 = 0.09465
    l3 = 0.0823
    theta1 = math.atan2(x, z)
    r = math.sqrt(x ** 2 + z ** 2)
    y_rel = y - l1
    d = math.sqrt(r ** 2 + y_rel ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    theta3 = math.pi / 2 - theta3
    beta = math.atan2(y_rel, r)
    gamma = math.asin(l3 * math.sin(math.pi - theta3) / d)
    theta2 = beta - gamma
    return (theta1, theta2, theta3)