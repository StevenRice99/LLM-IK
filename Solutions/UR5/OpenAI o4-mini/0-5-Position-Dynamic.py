import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed-form IK for the 6‑DOF chain, solving only the TCP position.
    Joint6 is redundant for position and held at zero.
    :param p: target (x, y, z)
    :return: (q1,q2,q3,q4,q5,q6) in radians
    """
    x_target, y_target, z_target = p
    q1 = math.atan2(-x_target, y_target)
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    x2 = c1 * x_target + s1 * y_target
    y2 = -s1 * x_target + c1 * y_target
    z2 = z_target
    y2 -= 0.13585
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    y_offset = -0.1197 + 0.093
    tcp_y = 0.0823

    def normalize(theta):
        while theta > math.pi:
            theta -= 2 * math.pi
        while theta < -math.pi:
            theta += 2 * math.pi
        return theta
    C = (y2 - y_offset) / tcp_y
    C = max(min(C, 1.0), -1.0)
    q5_cands = [math.acos(C), -math.acos(C)]
    psi = math.atan2(x2, z2)
    best = None
    for q5 in q5_cands:
        d = tcp_y * math.sin(q5)
        L_eff = math.hypot(L3, d)
        phi = math.atan2(d, L3)
        for T in (psi, psi + math.pi):
            S = T + phi
            W_x = x2 - L_eff * math.sin(T)
            W_z = z2 - L_eff * math.cos(T)
            r_w = math.hypot(W_x, W_z)
            if r_w < abs(L1 - L2) or r_w > L1 + L2:
                continue
            cos_q3 = (r_w * r_w - L1 * L1 - L2 * L2) / (2 * L1 * L2)
            cos_q3 = max(min(cos_q3, 1.0), -1.0)
            for sign in (+1, -1):
                q3 = sign * math.acos(cos_q3)
                delta = math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
                theta_w = math.atan2(W_x, W_z)
                q2 = theta_w - delta
                q4 = S - (q2 + q3)
                q6 = 0.0
                sol = (normalize(q1), normalize(q2), normalize(q3), normalize(q4), normalize(q5), normalize(q6))
                best = sol
                break
            if best:
                break
        if best:
            break
    if best is None:
        raise ValueError('No IK solution found for position {}'.format(p))
    return best