import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    joint1 = -math.atan2(-x, z)
    d = math.sqrt(x ** 2 + z ** 2)
    l1 = 0.425
    l2 = 0.39225
    l3 = 0.093
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = math.acos(cos_theta2)
    cos_theta3 = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
    theta3 = math.acos(cos_theta3)
    joint2 = theta2
    joint3 = theta3
    return (joint1, joint2, joint3)