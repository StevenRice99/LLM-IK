def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    L2 = 0.425
    L3 = 0.39225
    h = 0.13585
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    z_rel = z - h
    d_squared = r ** 2 + z_rel ** 2
    cos_theta3 = (d_squared - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = math.atan2(math.sqrt(1 - cos_theta3 ** 2), cos_theta3)
    beta = math.atan2(r, z_rel)
    gamma = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2 = beta - gamma
    return (theta1, theta2, theta3)