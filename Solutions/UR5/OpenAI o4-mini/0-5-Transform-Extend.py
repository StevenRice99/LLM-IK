def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‐form 6‑DOF IK for the URDF chain:
       Z–Y–Y–Y–Z–Y axes, lengths as given in DETAILS,
       with final tool_offset [0,0.0823,0] and yaw=+π/2.
    """
    import math
    import numpy as np
    L2y, L3y = (0.13585, 0.1197)
    L1, L2 = (0.425, 0.39225)
    L5, L6 = (0.093, 0.09465)
    L7 = 0.0823
    y_bar = L2y - L3y
    y_const2 = y_bar + L5
    px, py, pz = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_des = R_z @ R_y @ R_x

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    Rz_m90 = Rz(-math.pi / 2)
    Rxy = math.hypot(px, py)
    if Rxy < 1e-08:
        raise ValueError('degenerate XY position')
    arg = y_const2 / Rxy
    arg = max(-1.0, min(1.0, arg))
    alpha = math.asin(arg)
    theta = math.atan2(py, px)
    q1_cands = [theta - alpha, theta - (math.pi - alpha)]
    best = None
    best_err = 1000000000.0

    def cap(x):
        return max(-1.0, min(1.0, x))
    for q1 in q1_cands:
        c1, s1 = (math.cos(q1), math.sin(q1))
        Rz_n1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        p_bar = Rz_n1 @ np.array([px, py, pz])
        M = Rz_n1 @ R_des @ Rz_m90
        phi = math.atan2(M[2, 1], -M[0, 1])
        c5 = cap(M[1, 1])
        s5 = math.sqrt(max(0.0, 1.0 - c5 * c5))
        for sign5 in (+1.0, -1.0):
            q5 = math.atan2(sign5 * s5, c5)
            if abs(s5) < 1e-08:
                q6 = 0.0
            else:
                q6 = math.atan2(M[1, 2] / sign5, M[1, 0] / sign5)
            u = p_bar - np.array([0.0, L5, 0.0])
            d6 = np.array([-L6 * math.sin(q6) * math.sin(q5), L6 * math.sin(q6) * math.cos(q5), L6 * math.cos(q6)])
            d7 = np.array([-L7 * math.sin(q5), L7 * math.cos(q5), 0.0])
            u -= d6 + d7
            ux, uy, uz = u
            r2 = math.hypot(ux, uz)
            cos_q3 = cap((r2 * r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2))
            for sign3 in (+1.0, -1.0):
                q3 = sign3 * math.acos(cos_q3)
                q2 = math.atan2(ux, uz) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
                q4 = phi - (q2 + q3)
                x_calc = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
                z_calc = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
                err_pos = math.hypot(x_calc - ux, z_calc - uz) + abs(uy - y_bar)
                R_test = Rz(q1) @ Ry(q2) @ Ry(q3) @ Ry(q4) @ Rz(q5) @ Ry(q6) @ Rz(math.pi / 2)
                err_orient = np.linalg.norm(R_test - R_des, ord='fro')
                err = err_pos + err_orient
                if err < best_err:
                    best_err = err
                    best = (q1, q2, q3, q4, q5, q6)
    if best is None:
        raise ValueError('IK failure – no solution found')
    return best