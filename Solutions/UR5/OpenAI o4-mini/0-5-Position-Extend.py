import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‑form inverse kinematics for the 6‑DOF arm (Z–Y–Y–Y–Z–Y) to match TCP position p=(x,y,z).
    We fold the final TCP-offset (0.0823\xa0m along the local Y of joint\u20096) into the y-chain
    and then solve exactly as the 5‑DOF planar subchain, keeping θ5=θ6=0.
    """
    x_w, y_w, z_w = p
    d2_y = 0.13585
    d23_y = -0.1197
    d34_z = 0.425
    d45_y = 0.093
    d5e_z = 0.09465
    d_tcp_y = 0.0823
    y_chain = d23_y + d45_y + d_tcp_y
    L1 = d34_z
    L2 = 0.39225
    L3 = d5e_z
    delta = y_chain + d2_y
    r = math.hypot(x_w, y_w)
    arg = max(-1.0, min(1.0, delta / r))
    gamma = math.acos(arg)
    phi = math.atan2(-x_w, y_w)
    t1_cands = [phi + gamma, phi - gamma]
    best = (1000000000.0, 0.0, 0.0, 0.0, 0.0)
    for t1 in t1_cands:
        c1 = math.cos(t1)
        s1 = math.sin(t1)
        x2 = c1 * x_w + s1 * y_w
        y2 = -s1 * x_w + c1 * y_w - d2_y
        z2 = z_w
        psi = math.atan2(x2, z2)
        for T in (psi, psi + math.pi):
            xw = x2 - L3 * math.sin(T)
            zw = z2 - L3 * math.cos(T)
            rw2 = xw * xw + zw * zw
            cosb = (rw2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
            cosb = max(-1.0, min(1.0, cosb))
            for signb in (1.0, -1.0):
                t3 = signb * math.acos(cosb)
                phi_w = math.atan2(xw, zw)
                delta_w = math.atan2(L2 * math.sin(t3), L1 + L2 * math.cos(t3))
                t2 = phi_w - delta_w
                t4 = T - (t2 + t3)
                x_fk = L1 * math.sin(t2) + L2 * math.sin(t2 + t3) + L3 * math.sin(t2 + t3 + t4)
                z_fk = L1 * math.cos(t2) + L2 * math.cos(t2 + t3) + L3 * math.cos(t2 + t3 + t4)
                err = (x_fk - x2) ** 2 + (z_fk - z2) ** 2 + (y2 - y_chain) ** 2
                if err < best[0]:
                    best = (err, t1, t2, t3, t4)
    _, θ1, θ2, θ3, θ4 = best
    θ5 = 0.0
    θ6 = 0.0

    def norm(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
    return (norm(θ1), norm(θ2), norm(θ3), norm(θ4), norm(θ5), norm(θ6))