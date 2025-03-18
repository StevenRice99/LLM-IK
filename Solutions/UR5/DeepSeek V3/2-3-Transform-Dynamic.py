import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1 = math.atan2(x, z)
    revolute2_x = 0.39225 * math.sin(theta1)
    revolute2_z = 0.39225 * math.cos(theta1)
    tcp_x = x - revolute2_x
    tcp_z = z - revolute2_z
    theta2 = math.atan2(tcp_x, tcp_z) - theta1
    return (theta1, theta2)