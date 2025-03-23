def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed-form IK for the 6-DOF manipulator, enumerating principal solution branches
    and then choosing the one that best matches the requested (p, r). This avoids
    iterative or symbolic solvers and handles multiple branches explicitly.

    Robot geometry (from DETAILS):
      1) Revolute about Z at [0, 0, 0]
      2) Revolute about Y at [0, 0.13585, 0]
      3) Revolute about Y at [0, -0.1197, 0.425]
      4) Revolute about Y at [0, 0, 0.39225]
      5) Revolute about Z at [0, 0.093, 0]
      6) Revolute about Y at [0, 0, 0.09465]
      TCP) offset [0, 0.0823, 0] plus an intrinsic Rz(π/2).

    Approach:
      1) We follow a “decoupled” approach, first solving for (q1..q5) similarly to a 5-DOF subchain.
         This yields two possible q1 solutions (from the base XY-plane geometry) and two possible
         q3 solutions (±acos(·)) for the 2R sub-problem. We then compute q2, q4, q5 from those.
      2) For each such (q1..q5), we solve q6 by factoring out Rz(q1)*Ry(q2+q3+q4)*Rz(q5) from the
         desired orientation, leaving a pure rotation about Y (plus the known Rz(π/2) at the TCP).
         That yields one principal value from atan2, but we also consider q6 + π as a second branch
         for completeness. In total, we get up to 2×2×2=8 candidate solutions.
      3) We compute forward kinematics for each candidate and measure the pose error relative to
         (p, r). We return the candidate with the smallest error. This is purely closed-form.

    No reachability checks or joint range clips are implemented, as per instructions.
    """
    import math
    import numpy as np
    x, y, z = p
    roll, pitch, yaw = r

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def forward_kinematics(q1_val, q2_val, q3_val, q4_val, q5_val, q6_val):

        def make_T(dx, dy, dz, R_3x3):
            T_ = np.eye(4)
            T_[0:3, 0:3] = R_3x3
            T_[0, 3] = dx
            T_[1, 3] = dy
            T_[2, 3] = dz
            return T_
        T1 = make_T(0, 0, 0, Rz(q1_val))
        T2 = make_T(0, 0.13585, 0, Ry(q2_val))
        T3 = make_T(0, -0.1197, 0.425, Ry(q3_val))
        T4 = make_T(0, 0, 0.39225, Ry(q4_val))
        T5 = make_T(0, 0.093, 0, Rz(q5_val))
        T6 = make_T(0, 0, 0.09465, Ry(q6_val))
        T_tcp = make_T(0, 0.0823, 0, Rz(math.pi / 2))
        T_ = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T_tcp
        return T_
    T_des = np.eye(4)
    T_des[0, 3] = x
    T_des[1, 3] = y
    T_des[2, 3] = z
    R_des = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    T_des[0:3, 0:3] = R_des
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093

    def solve_5dof(q1_guess, sign_q3_plus):
        cq1 = math.cos(q1_guess)
        sq1 = math.sin(q1_guess)
        Rz_neg_q1 = np.array([[cq1, sq1, 0], [-sq1, cq1, 0], [0, 0, 1]])
        p_bar_ = Rz_neg_q1 @ np.array([x, y, z])
        px_bar, py_bar, pz_bar = p_bar_
        M_ = Rz_neg_q1 @ R_des
        phi_ = math.atan2(M_[0, 2], M_[2, 2])
        q5_ = math.atan2(M_[1, 0], M_[1, 1])
        px_adj = px_bar - L_tcp * math.sin(phi_)
        pz_adj = pz_bar - L_tcp * math.cos(phi_)
        r2_ = math.hypot(px_adj, pz_adj)
        cos_q3 = (r2_ ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        main_q3 = math.acos(cos_q3)
        if not sign_q3_plus:
            main_q3 = -main_q3
        q2_ = math.atan2(px_adj, pz_adj) - math.atan2(L2 * math.sin(main_q3), L1 + L2 * math.cos(main_q3))
        q4_ = phi_ - (q2_ + main_q3)
        return (q2_, main_q3, q4_, q5_)

    def q1_branches():
        r_xy = math.hypot(x, y)
        if r_xy < 1e-12:
            return [0.0, math.pi]
        ratio = y_const / r_xy
        ratio = max(-1.0, min(1.0, ratio))
        a_ = math.asin(ratio)
        theta_ = math.atan2(y, x)
        return [theta_ - a_, theta_ - (math.pi - a_)]

    def pose_error(Ta, Tb):
        dx = Ta[0, 3] - Tb[0, 3]
        dy = Ta[1, 3] - Tb[1, 3]
        dz = Ta[2, 3] - Tb[2, 3]
        pos_err = dx * dx + dy * dy + dz * dz
        Ra = Ta[0:3, 0:3]
        Rb = Tb[0:3, 0:3]
        Rdiff = Ra - Rb
        rot_err = np.sum(Rdiff ** 2)
        return pos_err + rot_err
    solutions = []
    q1_list = q1_branches()
    for q1cand in q1_list:
        for sign_q3 in [True, False]:
            try:
                q2_, q3_, q4_, q5_ = solve_5dof(q1cand, sign_q3)
            except ValueError:
                continue
            R_leftover = Rz(q1cand) @ Ry(q2_ + q3_ + q4_) @ Rz(q5_)
            M6 = R_leftover.T @ R_des
            M6p = M6 @ Rz(-math.pi / 2)
            q6_main = math.atan2(M6p[0, 2], M6p[2, 2])
            for add_pi in [0.0, math.pi]:
                q6_ = q6_main + add_pi
                T_ = forward_kinematics(q1cand, q2_, q3_, q4_, q5_, q6_)
                err_ = pose_error(T_, T_des)
                solutions.append((err_, (q1cand, q2_, q3_, q4_, q5_, q6_)))
    if not solutions:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    solutions.sort(key=lambda x: x[0])
    best_sol = solutions[0][1]
    return best_sol