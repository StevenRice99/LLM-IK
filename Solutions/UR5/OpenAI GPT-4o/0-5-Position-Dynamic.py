import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    a2 = 0.425
    a3 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    tcp_offset = 0.0823
    theta1 = math.atan2(-x, y)
    x_tcp = x
    y_tcp = y - tcp_offset
    z_tcp = z
    r = math.sqrt(x_tcp ** 2 + z_tcp ** 2)
    r_squared = r ** 2 - d1 ** 2
    if r_squared < 0:
        r_squared = 0
    r_prime = math.sqrt(r_squared)
    phi = math.atan2(z_tcp, x_tcp)
    phi_prime = math.atan2(d1, r_prime)
    cos_theta3 = (r_prime ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    cos_theta3 = max(-1, min(1, cos_theta3))
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta2 = phi - phi_prime - math.atan2(a3 * sin_theta3, a2 + a3 * cos_theta3)
    theta4 = 0
    theta5 = 0
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)