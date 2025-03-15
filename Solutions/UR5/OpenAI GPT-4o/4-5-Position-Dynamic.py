import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.0823
    adjusted_x = x
    adjusted_y = y - tcp_offset
    adjusted_z = z - 0.09465
    theta1 = math.atan2(y, x)
    r = math.sqrt(adjusted_x ** 2 + adjusted_y ** 2)
    theta2 = math.atan2(adjusted_z, r)
    return (theta1, theta2)