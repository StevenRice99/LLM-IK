import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    y_offset = -0.1197
    z_offset = 0.425
    tcp_length = 0.39225
    z_adjusted = z - tcp_length
    x_rel = x
    z_rel = z_adjusted - z_offset
    theta_2 = math.atan2(x_rel, z_rel)
    d = math.sqrt(x ** 2 + z_adjusted ** 2)
    theta_1 = math.atan2(y - y_offset, d)
    return (theta_1, theta_2)