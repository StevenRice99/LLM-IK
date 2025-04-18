def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import math
    x_tcp, y_tcp, z_tcp = p
    tcp_offset_y = 0.0823
    d2_y = 0.13585
    d3_y = -0.1197
    d3_z = 0.425
    d4_z = 0.39225
    d5_y = 0.093
    d6_z = 0.09465
    y_chain = d3_y + d5_y
    L1 = d3_z
    L2 = d4_z
    L3 = d6_z
    delta = y_chain + d2_y
    r = math.hypot(x_tcp, y_tcp - tcp_offset_y)
    candidates = []
    phi = math.atan2(-x_tcp, y_tcp - tcp_offset_y)
    try:
        arg = delta / r
        arg = max(-1.0, min(1.0, arg))
        gamma = math.acos(arg)
        candidates.append((phi + gamma, +1))
        candidates.append((phi - gamma, -1))
    except:
        pass
    candidates.append((math.atan2(y_tcp, x_tcp), +1))
    candidates.append((math.atan2(y_tcp, x_tcp) + math.pi, -1))
    for offset in [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi, -math.pi / 4, -math.pi / 2, -3 * math.pi / 4]:
        candidates.append((phi + offset, +1))
    best = (float('inf'), 0, 0, 0, 0, 0)
    for t1, sign in candidates:
        c1, s1 = (math.cos(t1), math.sin(t1))
        x2 = c1 * x_tcp + s1 * (y_tcp - tcp_offset_y)
        y2 = -s1 * x_tcp + c1 * (y_tcp - tcp_offset_y) - d2_y
        z2 = z_tcp
        psi = math.atan2(x2, z2)
        for T in (psi, psi + math.pi):
            xw = x2 - L3 * math.sin(T)
            zw = z2 - L3 * math.cos(T)
            rw2 = xw * xw + zw * zw
            if rw2 > (L1 + L2) ** 2 or rw2 < (L1 - L2) ** 2:
                continue
            cosb = (rw2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
            cosb = max(-1.0, min(1.0, cosb))
            for elbow_sign in (+1.0, -1.0):
                b = elbow_sign * math.acos(cosb)
                phi_w = math.atan2(xw, zw)
                delta_w = math.atan2(L2 * math.sin(b), L1 + L2 * math.cos(b))
                t2 = phi_w - delta_w
                t3 = b
                t4 = T - (t2 + t3)
                for t5 in (0, math.pi / 2, math.pi, -math.pi / 2, math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4):
                    x_6 = L1 * math.sin(t2) + L2 * math.sin(t2 + t3) + L3 * math.sin(t2 + t3 + t4)
                    z_6 = L1 * math.cos(t2) + L2 * math.cos(t2 + t3) + L3 * math.cos(t2 + t3 + t4)
                    y_6 = y_chain
                    c5, s5 = (math.cos(t5), math.sin(t5))
                    tcp_x = -tcp_offset_y * s5
                    tcp_y = tcp_offset_y * c5
                    x_tcp_calc = c1 * (x_6 + tcp_x) - s1 * (y_6 + tcp_y)
                    y_tcp_calc = s1 * (x_6 + tcp_x) + c1 * (y_6 + tcp_y) + d2_y
                    z_tcp_calc = z_6
                    err = (x_tcp_calc - x_tcp) ** 2 + (y_tcp_calc - y_tcp) ** 2 + (z_tcp_calc - z_tcp) ** 2
                    if err < best[0]:
                        best = (err, t1, t2, t3, t4, t5)
    if best[0] == float('inf'):
        return (0, 0, 0, 0, 0, 0)
    _, θ1, θ2, θ3, θ4, θ5 = best
    θ6 = 0.0

    def norm(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
    return (norm(θ1), norm(θ2), norm(θ3), norm(θ4), norm(θ5), θ6)