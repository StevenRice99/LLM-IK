import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.425
    d2 = 0.1197
    d3 = 0.39225
    tcp_offset = 0.093
    y_adjusted = y + tcp_offset
    theta1 = math.atan2(x, z)
    z2 = z - d1
    r = math.sqrt(x ** 2 + z2 ** 2)
    theta2 = math.atan2(-y_adjusted, r)
    theta3 = 0
    return (theta1, theta2, theta3)