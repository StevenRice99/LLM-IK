import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    _, pitch, _ = r
    theta1 = math.atan2(-x, y)
    x_prime = x * math.cos(theta1) + y * math.sin(theta1)
    y_prime = -x * math.sin(theta1) + y * math.cos(theta1)
    z_prime = z
    l1 = 0.13585
    l2_y = -0.1197
    l2_z = 0.425
    target_y = y_prime - l2_y
    target_z = z_prime - l2_z
    theta2 = math.atan2(target_z, target_y) - pitch
    return (theta1, theta2)