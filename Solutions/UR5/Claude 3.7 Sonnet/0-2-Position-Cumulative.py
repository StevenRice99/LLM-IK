import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    l1_y = 0.13585
    l2_y = -0.1197
    l2_z = 0.425
    l3_z = 0.39225
    theta1 = math.atan2(px, py)
    r_xy = math.sqrt(px ** 2 + py ** 2)
    r_xy_adj = r_xy - l1_y
    if r_xy > 0:
        ux = px / r_xy
        uy = py / r_xy
    else:
        ux, uy = (0, 0)
    x_in_j1 = px * math.cos(theta1) + py * math.sin(theta1)
    y_in_j1 = -px * math.sin(theta1) + py * math.cos(theta1)
    z_in_j1 = pz
    y_in_j1 = y_in_j1 - l1_y
    d = math.sqrt(y_in_j1 ** 2 + z_in_j1 ** 2)
    l2_eff = math.sqrt(l2_y ** 2 + l2_z ** 2)
    l3_eff = l3_z
    alpha2 = math.atan2(l2_z, l2_y)
    cos_beta = (d ** 2 - l2_eff ** 2 - l3_eff ** 2) / (2 * l2_eff * l3_eff)
    cos_beta = max(min(cos_beta, 1.0), -1.0)
    beta = math.acos(cos_beta)
    theta3 = math.pi - beta
    gamma = math.atan2(z_in_j1, y_in_j1)
    cos_delta = (l2_eff ** 2 + d ** 2 - l3_eff ** 2) / (2 * l2_eff * d)
    cos_delta = max(min(cos_delta, 1.0), -1.0)
    delta = math.acos(cos_delta)
    theta2 = gamma - delta - alpha2
    return (theta1, theta2, theta3)