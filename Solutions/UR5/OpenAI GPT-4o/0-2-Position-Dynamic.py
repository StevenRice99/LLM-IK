import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    d = math.sqrt(x ** 2 + y ** 2)
    z_offset = z - 0.13585
    l1 = 0.425
    l2 = 0.39225
    r = math.sqrt(d ** 2 + z_offset ** 2)
    cos_theta3 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    alpha = math.atan2(z_offset, d)
    beta = math.atan2(l2 * sin_theta3, l1 + l2 * cos_theta3)
    theta2 = alpha - beta
    theta1 = -theta1
    return (theta1, theta2, theta3)