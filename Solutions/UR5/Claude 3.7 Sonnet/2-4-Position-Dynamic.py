import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    theta3 = 0.0
    theta1 = math.atan2(x, z)
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    x_1 = c1 * x + s1 * z
    z_1 = -s1 * x + c1 * z
    d_xz = math.sqrt(x_1 ** 2 + (z_1 - l1) ** 2)
    phi = math.atan2(x_1, z_1 - l1)
    d_tcp = l3
    d_23 = math.sqrt(l2 ** 2 + d_tcp ** 2)
    cos_alpha = (d_xz ** 2 + d_23 ** 2 - d_tcp ** 2) / (2 * d_xz * d_23)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    theta2 = phi - alpha
    return (theta1, theta2, theta3)