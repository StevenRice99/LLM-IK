import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    tcp_offset = [0, 0.0823, 0]
    adjusted_p = (p[0] - tcp_offset[0], p[1] - tcp_offset[1], p[2] - tcp_offset[2])
    x, y, z = adjusted_p
    link_length = 0.09465
    theta2 = math.atan2(z, link_length)
    theta1 = math.atan2(y, x)
    return (theta1, theta2)