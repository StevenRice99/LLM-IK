import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x_target, y_target, z_target = p
    tcp_offset = [0, -0.1197, 0.425]
    x_tcp = x_target - tcp_offset[0]
    y_tcp = y_target - tcp_offset[1]
    z_tcp = z_target - tcp_offset[2]
    theta_1 = math.atan2(-x_tcp, y_tcp)
    y_adjusted = y_tcp - 0.13585
    theta_2 = -math.atan2(-x_tcp, z_tcp)
    return (theta_1, theta_2)