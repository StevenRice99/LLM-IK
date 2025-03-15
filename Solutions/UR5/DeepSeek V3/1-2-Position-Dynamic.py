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
    theta1 = math.atan2(x, z)
    d = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    if x < 0:
        theta1 = theta1 + math.pi
    if z < 0:
        theta2 = -theta2
    return (theta1, theta2)