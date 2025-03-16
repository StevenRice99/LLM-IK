def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import math
    x, y, z = p
    L3_TCP = 0.39225
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    L2_3 = math.sqrt(0.425 ** 2 + 0.1197 ** 2)
    alpha = math.atan2(0.1197, 0.425)
    L1_2 = 0.13585
    theta1 = math.atan2(y, x)
    r_proj = math.sqrt(x ** 2 + y ** 2)
    d_xy = math.sqrt(x ** 2 + y ** 2)
    theta1 = math.atan2(y, x)
    d = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta1 = math.atan2(y, x)
    x_proj = math.sqrt(x ** 2 + y ** 2)
    z_proj = z
    L1 = 0.13585
    L2 = 0.425
    L2_y = -0.1197
    L3 = 0.39225
    theta1 = math.atan2(y, x)
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    L2_eff = math.sqrt(L2 ** 2 + L2_y ** 2)
    phi = math.atan2(-L2_y, L2)
    cos_theta3 = (r ** 2 + (z - L1) ** 2 - L2_eff ** 2 - L3 ** 2) / (2 * L2_eff * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    beta = math.atan2(L3 * sin_theta3, L2_eff + L3 * cos_theta3)
    gamma = math.atan2(z - L1, r)
    theta2 = gamma - beta + phi
    theta3 = theta3 - phi
    return (theta1, theta2, theta3)