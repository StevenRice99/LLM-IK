import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    x_prime = x * math.cos(theta1) + z * math.sin(theta1)
    z_prime = -x * math.sin(theta1) + z * math.cos(theta1)
    x_tcp = x_prime
    y_tcp = y + 0.1197 - 0.093
    z_tcp = z_prime - 0.425
    d = math.sqrt(x_tcp ** 2 + z_tcp ** 2)
    a = 0.39225
    b = 0.425
    cos_theta3 = (a ** 2 + b ** 2 - d ** 2) / (2 * a * b)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    theta2 = math.atan2(z_tcp, x_tcp) - math.atan2(b * sin_theta3, a + b * cos_theta3)
    theta2 = -theta2
    theta3 = -theta3
    return (theta1, theta2, theta3)