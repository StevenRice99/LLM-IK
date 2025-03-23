def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r" for the full 6-DOF chain.

    Closed-form analytical solution outline:
      1) Solve for q1 from the base-plane geometry (matching y_const).
      2) Define M_6 = Rz(-q1) * R_des * Rz(-π/2), where R_des = Rz(yaw)*Ry(pitch)*Rx(roll).
      3) Factor M_6 as Ry(φ)*Rz(q5)*Ry(q6).  From that, extract φ = q2 + q3 + q4, then q5, q6.
      4) Use the planar 2R approach (similar to the existing 5-DOF code) to solve for q2, q3, q4 from the position, subtracting out a total offset. 
      5) Return (q1, q2, q3, q4, q5, q6).

    :param p: The position to reach as (x, y, z).
    :param r: The orientation (roll, pitch, yaw) in radians, i.e. (rx, ry, rz).
    :return: (q1, q2, q3, q4, q5, q6), each in radians.
    """
    import math
    import numpy as np
    import sympy
    L1 = 0.425
    L2 = 0.39225
    L_tcp_5dof = 0.09465
    L_tcp_total = L_tcp_5dof + 0.0823
    y_const = 0.13585 - 0.1197 + 0.093
    px, py, pz = p
    roll, pitch, yaw = r
    r_xy = math.sqrt(px ** 2 + py ** 2)
    theta = math.atan2(py, px)
    ratio = y_const / (r_xy if r_xy != 0 else 1e-12)
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_candidate1 = theta - a
    q1_candidate2 = theta - (math.pi - a)

    def compute_M(q1_val):
        """
        M_6 = Rz(-q1) * R_des * Rz(-π/2)
        """
        cq1 = math.cos(q1_val)
        sq1 = math.sin(q1_val)
        Rz_neg_q1 = np.array([[cq1, sq1, 0], [-sq1, cq1, 0], [0, 0, 1]])
        cx, sx = (math.cos(roll), math.sin(roll))
        cy, sy = (math.cos(pitch), math.sin(pitch))
        cz, sz = (math.cos(yaw), math.sin(yaw))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_des = Rz @ Ry @ Rx
        Rz_neg_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        return Rz_neg_q1 @ R_des @ Rz_neg_90
    M1 = compute_M(q1_candidate1)
    M2 = compute_M(q1_candidate2)
    err1 = abs(M1[1, 2])
    err2 = abs(M2[1, 2])
    q1 = q1_candidate1 if err1 <= err2 else q1_candidate2
    M_6 = compute_M(q1)
    phi_sym, q5_sym, q6_sym = sympy.symbols('phi q5 q6', real=True)

    def Ry_sym(a):
        return sympy.Matrix([[sympy.cos(a), 0, sympy.sin(a)], [0, 1, 0], [-sympy.sin(a), 0, sympy.cos(a)]])

    def Rz_sym(a):
        return sympy.Matrix([[sympy.cos(a), -sympy.sin(a), 0], [sympy.sin(a), sympy.cos(a), 0], [0, 0, 1]])
    S_sym = Ry_sym(phi_sym) * Rz_sym(q5_sym) * Ry_sym(q6_sym)
    M_6_sym = sympy.Matrix(M_6)
    eqs = []
    for i in range(3):
        for j in range(3):
            eqs.append(sympy.Eq(S_sym[i, j], M_6_sym[i, j]))
    sol = sympy.solve(eqs, [phi_sym, q5_sym, q6_sym], dict=True, real=True)
    candidate_solutions = []
    for s in sol:
        candidate_solutions.append((float(s[phi_sym]), float(s[q5_sym]), float(s[q6_sym])))

    def orientation_err(phi_val, q5_val, q6_val):
        test_mat = np.array(Ry_sym(phi_val) * Rz_sym(q5_val) * Ry_sym(q6_val)).astype(float)
        return np.linalg.norm(test_mat - M_6)
    best = None
    best_error = 1000000000.0
    for phi_c, q5_c, q6_c in candidate_solutions:
        err_c = orientation_err(phi_c, q5_c, q6_c)
        if err_c < best_error:
            best_error = err_c
            best = (phi_c, q5_c, q6_c)
    if best is None:
        best = (0.0, 0.0, 0.0)
    phi, q5, q6 = best
    cq1 = math.cos(q1)
    sq1 = math.sin(q1)
    Rz_neg_q1 = np.array([[cq1, sq1, 0], [-sq1, cq1, 0], [0, 0, 1]])
    p_bar = Rz_neg_q1 @ np.array([px, py, pz])
    p_bar_x, _, p_bar_z = p_bar
    P_x = p_bar_x - L_tcp_total * math.sin(phi)
    P_z = p_bar_z - L_tcp_total * math.cos(phi)

    def planar_solution(q3_val):
        q2_val = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q4_val = phi - (q2_val + q3_val)
        xx = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L_tcp_total * math.sin(phi)
        zz = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L_tcp_total * math.cos(phi)
        return (q2_val, q4_val, math.hypot(xx - p_bar_x, zz - p_bar_z))
    r2 = math.hypot(P_x, P_z)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3A = math.acos(cos_q3)
    q3B = -q3A
    q2A, q4A, errA = planar_solution(q3A)
    q2B, q4B, errB = planar_solution(q3B)
    if errA <= errB:
        q3 = q3A
        q2 = q2A
        q4 = q4A
    else:
        q3 = q3B
        q2 = q2B
        q4 = q4B
    return (q1, q2, q3, q4, q5, q6)