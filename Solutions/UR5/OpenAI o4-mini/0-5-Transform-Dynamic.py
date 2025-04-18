def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    6‑DOF analytical IK for the chain with:
      • Revolute1 about Z
      • Revolute2–4 about Y
      • Revolute5 about Z
      • Revolute6 about Y
      • TCP offset: translate [0,0.0823,0], then Rz(pi/2)

    Major fixes vs. earlier attempt:
      (1) Add TCP‐Y offset into y_const
      (2) Solve 5‑DOF against R_des5 = R_des_input @ Rz(-pi/2)
      (3) At the end recover q6 by Rz(–π/2)·R_des_input
    """
    import math
    import numpy as np
    L1 = 0.425
    L2 = 0.39225
    L6z = 0.09465
    d_tcp_y = 0.0823
    y_const = 0.13585 - 0.1197 + 0.093 + d_tcp_y
    px, py, pz = p
    roll, pitch, yaw = r
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_des_input = Rz @ Ry @ Rx
    psi = 1.570796325
    c_psi = math.cos(psi)
    s_psi = math.sin(psi)
    Rz_tcp_inv = np.array([[c_psi, s_psi, 0], [-s_psi, c_psi, 0], [0, 0, 1]])
    R_des5 = R_des_input @ Rz_tcp_inv
    r_xy = math.hypot(px, py)
    theta = math.atan2(py, px)
    ratio = y_const / r_xy
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_c1 = theta - a
    q1_c2 = theta - (math.pi - a)

    def M_of(q1):
        c1 = math.cos(q1)
        s1 = math.sin(q1)
        Rz_neg = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        return Rz_neg @ R_des5
    M1 = M_of(q1_c1)
    M2 = M_of(q1_c2)
    q1 = q1_c1 if abs(M1[1, 2]) <= abs(M2[1, 2]) else q1_c2
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    Rz_neg1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    p_bar = Rz_neg1 @ np.array([px, py, pz])
    x_b, y_b, z_b = p_bar
    M = Rz_neg1 @ R_des5
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = math.atan2(M[1, 0], M[1, 1])
    Px = x_b - L6z * math.sin(phi)
    Pz = z_b - L6z * math.cos(phi)
    r2 = math.hypot(Px, Pz)
    cos_q3 = (r2 * r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3a = math.acos(cos_q3)
    q3b = -q3a

    def plan(q3):
        q2 = math.atan2(Px, Pz) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
        q4 = phi - (q2 + q3)
        x_r = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) + L6z * math.sin(phi)
        z_r = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) + L6z * math.cos(phi)
        err = math.hypot(x_r - x_b, z_r - z_b)
        return (q2, q4, err)
    q2a, q4a, ea = plan(q3a)
    q2b, q4b, eb = plan(q3b)
    if ea <= eb:
        q2, q3, q4 = (q2a, q3a, q4a)
    else:
        q2, q3, q4 = (q2b, q3b, q4b)
    c_phi = math.cos(q2 + q3 + q4)
    s_phi = math.sin(q2 + q3 + q4)
    Ry_phi = np.array([[c_phi, 0, s_phi], [0, 1, 0], [-s_phi, 0, c_phi]])
    c5 = math.cos(q5)
    s5 = math.sin(q5)
    Rz5 = np.array([[c5, -s5, 0], [s5, c5, 0], [0, 0, 1]])
    Rz1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    R_pred5 = Rz1 @ (Ry_phi @ Rz5)
    R6 = R_pred5.T @ (R_des_input @ Rz_tcp_inv)
    q6 = math.atan2(R6[0, 2], R6[2, 2])
    return (q1, q2, q3, q4, q5, q6)