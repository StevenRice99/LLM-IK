import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.39225
    d2 = 0.093
    d3 = 0.09465
    wc_z = z - d3
    wc_x = x
    wc_y = y
    theta1 = math.atan2(wc_x, wc_z)
    r = math.sqrt(wc_x ** 2 + wc_z ** 2)
    s = wc_y - d1
    L = math.sqrt(r ** 2 + s ** 2)
    cos_theta2 = (d2 ** 2 + L ** 2 - d1 ** 2) / (2 * d2 * L)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(s, r) - math.atan2(sin_theta2, cos_theta2)
    theta3 = 0
    return (theta1, theta2, theta3)