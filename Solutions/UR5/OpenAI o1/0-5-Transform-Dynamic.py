def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    A more exhaustive closed-form-based inverse kinematics solution for this 6–DOF robot, 
    which systematically attempts ±2π “wraps” on each joint to find the best match to the 
    requested pose. This approach helps ensure that the correct branch is found even if 
    the basic geometry solution lands in a different rotation branch, or if large rotation 
    angles are needed.

    Procedure:
      1) Solve an initial set of angles (q1..q6) using the same geometric decomposition 
         as the 5–DOF subchain plus an extra joint. This yields a 'principal' solution. 
         Because of the planar subchain, we also get a sign-flip possibility for q3. 
         So we produce up to two solutions from the subchain approach (q3 = ±acos(...)).
      2) For each candidate from step (1), generate all 3^6 = 729 variants by adding 
         {0, +2π, -2π} to each joint angle.  (We also incorporate any 2π wrap for q3 
         from step (1).)
      3) Compute the forward kinematics for each variant and compare to the desired 
         pose (both position and orientation). 
      4) Return the variant that minimizes the overall pose error.

    Because the problem states that all inputs are reachable and joint limits are ±2π, 
    we do not add additional checks for unreachable or out-of-bounds angles.

    Args:
      p (float,float,float): Desired TCP position [x, y, z].
      r (float,float,float): Desired TCP roll–pitch–yaw [rx, ry, rz], in radians.

    Returns:
      (q1, q2, q3, q4, q5, q6) in radians, chosen to minimize final pose error.
    """
    import math
    import numpy as np

    def Rx(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], dtype=float)

    def Ry(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=float)

    def Rz(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=float)

    def rot3x3_to_4x4(R):
        M = np.eye(4)
        M[0:3, 0:3] = R
        return M

    def translate(x, y, z):
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        return T

    def forward_kin(j1, j2, j3, j4, j5, j6):
        """
        Returns a 4x4 transform from base to TCP, given the 6 joint angles. 
        This matches the "DETAILS" table plus the final TCP orientation offset.
        """
        T1 = rot3x3_to_4x4(Rz(j1))
        T2 = translate(0, 0.13585, 0) @ rot3x3_to_4x4(Ry(j2))
        T3 = translate(0, -0.1197, 0.425) @ rot3x3_to_4x4(Ry(j3))
        T4 = translate(0, 0, 0.39225) @ rot3x3_to_4x4(Ry(j4))
        T5 = translate(0, 0.093, 0) @ rot3x3_to_4x4(Rz(j5))
        T6 = translate(0, 0, 0.09465) @ rot3x3_to_4x4(Ry(j6))
        T_tcp = translate(0, 0, 0.0823) @ rot3x3_to_4x4(Rz(math.pi / 2))
        T_result = np.eye(4)
        for Ti in [T1, T2, T3, T4, T5, T6, T_tcp]:
            T_result = T_result @ Ti
        return T_result

    def pose_error(Ta, Tb):
        """
        Returns a scalar measure of difference between two 4x4 transforms:
        position error (Euclidean distance) + orientation difference (Frobenius norm).
        """
        dp = Ta[0:3, 3] - Tb[0:3, 3]
        err_pos = np.linalg.norm(dp)
        Ra = Ta[0:3, 0:3]
        Rb = Tb[0:3, 0:3]
        err_rot = np.linalg.norm(Ra - Rb, 'fro')
        return err_pos + err_rot
    p_x, p_y, p_z = p
    roll, pitch, yaw = r
    R_des = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    T_des = np.eye(4)
    T_des[:3, :3] = R_des
    T_des[0, 3] = p_x
    T_des[1, 3] = p_y
    T_des[2, 3] = p_z
    L1 = 0.425
    L2 = 0.39225
    L_tcp_5dof = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    R_des_5dof = R_des @ Rz(-math.pi / 2)
    r_xy = math.sqrt(p_x ** 2 + p_y ** 2) + 1e-14
    theta = math.atan2(p_y, p_x)
    ratio = max(-1.0, min(1.0, y_const / r_xy))
    a_ = math.asin(ratio)
    q1_candidates = [theta - a_, theta - (math.pi - a_)]

    def M_for_q1(q1v):
        c = math.cos(q1v)
        s = math.sin(q1v)
        Rz_nq1 = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=float)
        return Rz_nq1 @ R_des_5dof

    def pick_q1(q1cands):
        best_q1 = None
        best_err = 1000000000.0
        for q1v in q1cands:
            Mtest = M_for_q1(q1v)
            e = abs(Mtest[1, 2])
            if e < best_err:
                best_err = e
                best_q1 = q1v
        return best_q1
    q1_main = pick_q1(q1_candidates)
    M = M_for_q1(q1_main)
    phi = math.atan2(M[0, 2], M[2, 2])
    q5_main = math.atan2(M[1, 0], M[1, 1])
    p_vec = np.array([p_x, p_y, p_z], dtype=float)
    c1, s1 = (math.cos(q1_main), math.sin(q1_main))
    Rz_neg_q1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]], dtype=float)
    p_bar = Rz_neg_q1 @ p_vec
    p_bx, _, p_bz = p_bar
    P_x = p_bx - L_tcp_5dof * math.sin(phi)
    P_z = p_bz - L_tcp_5dof * math.cos(phi)
    r2 = math.sqrt(P_x ** 2 + P_z ** 2) + 1e-14
    cos_q3 = (r2 * r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_A = math.acos(cos_q3)
    q3_B = -q3_A

    def planar_subchain(q3v):
        q2v = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3v), L1 + L2 * math.cos(q3v))
        q4v = phi - (q2v + q3v)
        return (q2v, q3v, q4v)
    subchain_candidates = []
    for q3cand in [q3_A, q3_B]:
        q2cand, _, q4cand = planar_subchain(q3cand)
        subchain_candidates.append((q1_main, q2cand, q3cand, q4cand, q5_main))

    def compute_q6(q1v, q5v, phi, M):
        Ry_neg_phi = np.array([[math.cos(phi), 0, -math.sin(phi)], [0, 1, 0], [math.sin(phi), 0, math.cos(phi)]], dtype=float)
        Rz_neg_q5 = np.array([[math.cos(-q5v), math.sin(-q5v), 0], [-math.sin(-q5v), math.cos(-q5v), 0], [0, 0, 1]], dtype=float)
        M1 = Ry_neg_phi @ M
        M2 = Rz_neg_q5 @ M1
        return math.atan2(M2[0, 2], M2[2, 2])
    base_solutions = []
    for q1_b, q2_b, q3_b, q4_b, q5_b in subchain_candidates:
        M_b = M_for_q1(q1_b)
        q6_b = compute_q6(q1_b, q5_b, phi, M_b)
        base_solutions.append((q1_b, q2_b, q3_b, q4_b, q5_b, q6_b))

    def angle_variants(val):
        return [val, val + 2 * math.pi, val - 2 * math.pi]
    best_sol = None
    best_err = 1000000000.0
    for q_base in base_solutions:
        q1_0, q2_0, q3_0, q4_0, q5_0, q6_0 = q_base
        for Q1 in angle_variants(q1_0):
            for Q2 in angle_variants(q2_0):
                for Q3 in angle_variants(q3_0):
                    for Q4 in angle_variants(q4_0):
                        for Q5 in angle_variants(q5_0):
                            for Q6 in angle_variants(q6_0):
                                T_fk = forward_kin(Q1, Q2, Q3, Q4, Q5, Q6)
                                e = pose_error(T_fk, T_des)
                                if e < best_err:
                                    best_err = e
                                    best_sol = (Q1, Q2, Q3, Q4, Q5, Q6)
    return best_sol