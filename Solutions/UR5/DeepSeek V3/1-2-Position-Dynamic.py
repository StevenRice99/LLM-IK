import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    d = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(z, x) - math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    if x < 0:
        theta1 += math.pi
    return (theta1, theta2)