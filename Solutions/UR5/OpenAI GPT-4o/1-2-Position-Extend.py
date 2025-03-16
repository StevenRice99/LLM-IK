import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x_target, y_target, z_target = p
    l1 = 0.425
    l2 = 0.39225
    r = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(z_target, x_target)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = phi - math.atan2(k2, k1)
    if x_target < 0:
        theta1 += math.pi
    return (theta1, theta2)