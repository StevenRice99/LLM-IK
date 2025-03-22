import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    y_offset = -0.0267
    x_rot = x * math.cos(-theta1) - y * math.sin(-theta1)
    y_rot = x * math.sin(-theta1) + y * math.cos(-theta1)
    x_planar = math.hypot(x, y)
    z_planar = z

    def fk_planar(theta1, theta2, theta3):
        x_fk = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2) + L3 * math.sin(theta1 + theta2 + theta3)
        z_fk = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2) + L3 * math.cos(theta1 + theta2 + theta3)
        return (x_fk, y_offset, z_fk)
    psi = math.atan2(x_planar, z_planar)
    T_options = [psi, psi + math.pi]
    candidates = []
    for T in T_options:
        x_w = x_planar - L3 * math.sin(T)
        z_w = z_planar - L3 * math.cos(T)
        r_w = math.sqrt(x_w ** 2 + z_w ** 2)
        cos_beta = (r_w ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        for sign in (1, -1):
            beta = sign * math.acos(cos_beta)
            phi_w = math.atan2(x_w, z_w)
            delta = math.atan2(L2 * math.sin(beta), L1 + L2 * math.cos(beta))
            theta2_candidate = phi_w - delta
            theta3_candidate = beta
            theta4_candidate = T - (theta2_candidate + theta3_candidate)
            x_fk, _, z_fk = fk_planar(theta2_candidate, theta3_candidate, theta4_candidate)
            err = math.hypot(x_fk - x_planar, z_fk - z_planar)
            candidates.append((err, theta2_candidate, theta3_candidate, theta4_candidate))
    best = min(candidates, key=lambda tup: tup[0])
    theta2, theta3, theta4 = (best[1], best[2], best[3])
    theta5 = 0.0

    def normalize(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    theta1 = normalize(theta1)
    theta2 = normalize(theta2)
    theta3 = normalize(theta3)
    theta4 = normalize(theta4)
    theta5 = normalize(theta5)
    return (theta1, theta2, theta3, theta4, theta5)