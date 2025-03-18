import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = math.atan2(x, z)
    x_prime = math.cos(-theta1) * x - math.sin(-theta1) * z
    z_prime = math.sin(-theta1) * x + math.cos(-theta1) * z
    z_prime -= 0.39225
    theta2 = pitch - theta1
    if theta2 > math.pi:
        theta2 -= 2 * math.pi
    elif theta2 < -math.pi:
        theta2 += 2 * math.pi
    return (theta1, theta2)