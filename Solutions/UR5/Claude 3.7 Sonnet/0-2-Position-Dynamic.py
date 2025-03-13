def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import math
    import numpy as np
    px, py, pz = p
    theta1 = math.atan2(px, py)
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    x_prime = c1 * px + s1 * py
    y_prime = -s1 * px + c1 * py
    z_prime = pz
    y_prime -= 0.13585
    y_offset_3 = -0.1197
    y_target = y_prime - y_offset_3
    l1 = 0.425
    l2 = 0.39225
    r = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta3 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = math.acos(cos_theta3)
    beta = math.atan2(z_prime, x_prime)
    gamma = math.atan2(l2 * math.sin(theta3), l1 + l2 * math.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)