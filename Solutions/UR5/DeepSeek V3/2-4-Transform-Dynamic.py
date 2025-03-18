import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1 = math.atan2(x, z)
    x_prime = x * math.cos(theta1) + z * math.sin(theta1)
    z_prime = -x * math.sin(theta1) + z * math.cos(theta1)
    d = math.sqrt(x_prime ** 2 + (z_prime - 0.39225) ** 2)
    a = 0.093
    b = 0.09465
    cos_theta2 = (a ** 2 + d ** 2 - b ** 2) / (2 * a * d)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    theta3 = rz - (theta1 + theta2)
    return (theta1, theta2, theta3)