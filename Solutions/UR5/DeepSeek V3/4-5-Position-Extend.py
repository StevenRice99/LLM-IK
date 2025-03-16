import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l2 = 0.09465
    l_tcp = 0.0823
    d = math.sqrt(x ** 2 + y ** 2)
    cos_theta2 = (d ** 2 + z ** 2 - l2 ** 2 - l_tcp ** 2) / (2 * l2 * l_tcp)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(y, x)
    return (theta1, theta2)