import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed–form inverse kinematics for the 5‑DOF arm (Z–Y–Y–Y–Z) to reach TCP at p=(x,y,z).
    Link offsets (in local frames):
      • d2 = [0,   0.13585, 0]
      • d23= [0,  −0.1197,  0.425]
      • d34= [0,   0,       0.39225]
      • d45= [0,   0.093,   0]
      • d5E= [0,   0,       0.09465]  (E = end‑effector)
    We first pick θ1 so that in joint‑2’s frame the Y–coordinate of the target
    exactly matches the constant offset y₂ = −0.1197+0.093=−0.0267.
    Then we solve joints 2–4 as a 3R planar chain in that frame.
    """
    x_w, y_w, z_w = p
    d2_y = 0.13585
    y_chain = -0.1197 + 0.093
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    delta = y_chain + d2_y
    r = math.hypot(x_w, y_w)
    phi = math.atan2(-x_w, y_w)
    arg = delta / r
    arg = max(-1.0, min(1.0, arg))
    gamma = math.acos(arg)
    t1_cands = [phi + gamma, phi - gamma]
    best = (1000000000.0, 0, 0, 0, 0)
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
            for sign in (+1.0, -1.0):
                b = sign * math.acos(cosb)
                phi_w = math.atan2(xw, zw)
                delta_w = math.atan2(L2 * math.sin(b), L1 + L2 * math.cos(b))
                t2 = phi_w - delta_w
                t3 = b
                t4 = T - (t2 + t3)
                x_fk = L1 * math.sin(t2) + L2 * math.sin(t2 + t3) + L3 * math.sin(t2 + t3 + t4)
                z_fk = L1 * math.cos(t2) + L2 * math.cos(t2 + t3) + L3 * math.cos(t2 + t3 + t4)
                err2 = (x_fk - x2) ** 2 + (z_fk - z2) ** 2 + (y2 - y_chain) ** 2
                if err2 < best[0]:
                    best = (err2, t1, t2, t3, t4)
    _, θ1, θ2, θ3, θ4 = best
    θ5 = 0.0

    def norm(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a
    return (norm(θ1), norm(θ2), norm(θ3), norm(θ4), norm(θ5))