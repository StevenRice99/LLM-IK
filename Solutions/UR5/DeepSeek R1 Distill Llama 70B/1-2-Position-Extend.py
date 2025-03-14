import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    joint2_x, joint2_y, joint2_z = (0.0, -0.1197, 0.425)
    tcp_offset_z = 0.39225
    tcp_x = x - joint2_x
    tcp_y = y - joint2_y
    tcp_z = z - joint2_z - tcp_offset_z
    d = math.sqrt(tcp_x ** 2 + tcp_z ** 2)
    if d == 0:
        return (0.0, 0.0)
    theta1 = math.atan2(tcp_x, tcp_z)
    a1 = 0.434
    a2 = 0.39225
    cos_theta2 = (a1 ** 2 + a2 ** 2 - d ** 2) / (2 * a1 * a2)
    epsilon = 1e-08
    cos_theta2 = max(min(cos_theta2, 1.0 - epsilon), -1.0 + epsilon)
    theta2 = math.acos(cos_theta2)
    return (theta1, theta2)