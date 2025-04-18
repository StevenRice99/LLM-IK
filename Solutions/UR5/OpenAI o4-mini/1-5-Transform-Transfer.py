import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed–form IK for the 5‑DOF arm:
      q1,q2,q3 about Y; q4 about Z; q5 about Y; then a fixed Rz(+90deg) to the TCP.
    We extract:
      • q4 from the y–equation (two solutions),
      • S = q1+q2+q3 and q5 directly from R_target,
      • then solve the planar 2R for q1,q2 (two elbow branches) and q3 = S-(q1+q2).
    Finally, we pick the branch whose full FK (pos+orient) best matches the target.
    """
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    y_off = -0.1197 + 0.093
    tcp_y = 0.0823
    x_t, y_t, z_t = p
    roll, pitch, yaw = r

    def clamp(v, lo=-1.0, hi=1.0):
        return hi if v > hi else lo if v < lo else v

    def normalize(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]

    def mat_mult(A, B):
        """3×3 product A·B."""
        return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j] for j in range(3)] for i in range(3)]

    def transpose(A):
        return [[A[j][i] for j in range(3)] for i in range(3)]
    R_target = mat_mult(Rz(yaw), mat_mult(Ry(pitch), Rx(roll)))
    R_tcp_inv = Rz(-math.pi / 2)
    M = mat_mult(R_target, R_tcp_inv)
    M01 = M[0][1]
    M21 = M[2][1]
    M10 = M[1][0]
    M12 = M[1][2]
    S = math.atan2(M21, -M01)
    q5 = math.atan2(M12, -M10)
    C = clamp((y_t - y_off) / tcp_y)
    q4_cands = [math.acos(C), -math.acos(C)]
    best_cost = float('inf')
    best_sol = None
    for q4 in q4_cands:
        s4 = math.sin(q4)
        d = tcp_y * s4
        L_eff = math.hypot(L3, d)
        phi = math.atan2(d, L3)
        T = S - phi
        W_x = x_t - L_eff * math.sin(T)
        W_z = z_t - L_eff * math.cos(T)
        r_w = math.hypot(W_x, W_z)
        if r_w > L1 + L2 or r_w < abs(L1 - L2):
            continue
        cos_q2 = clamp((r_w * r_w - L1 * L1 - L2 * L2) / (2 * L1 * L2))
        for sign in (+1, -1):
            q2 = sign * math.acos(cos_q2)
            num = L2 * math.sin(q2)
            den = L1 + L2 * math.cos(q2)
            delta = math.atan2(num, den)
            theta = math.atan2(W_x, W_z)
            q1 = theta - delta
            q3 = S - (q1 + q2)
            S123 = q1 + q2 + q3
            d_ = tcp_y * math.sin(q4)
            x_fk = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(S123) - d_ * math.cos(S123)
            z_fk = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(S123) + d_ * math.sin(S123)
            y_fk = y_off + tcp_y * math.cos(q4)
            pos_err = math.hypot(x_fk - x_t, y_fk - y_t, z_fk - z_t)
            R1 = mat_mult(Ry(q5), Rz(math.pi / 2))
            R2 = mat_mult(Rz(q4), R1)
            R_fk = mat_mult(Ry(S123), R2)
            dR = mat_mult(transpose(R_fk), R_target)
            tr = dR[0][0] + dR[1][1] + dR[2][2]
            ang_err = math.acos(clamp((tr - 1.0) / 2.0))
            cost = pos_err + ang_err
            if cost < best_cost:
                best_cost = cost
                best_sol = (q1, q2, q3, q4, q5)
    if best_sol is None:
        raise ValueError('No IK solution found for the given pose')
    return tuple((normalize(q) for q in best_sol))