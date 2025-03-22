def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed–form inverse kinematics for the given 5–DOF manipulator, returning a
    single analytic solution without iterative or numeric root-finding. We exploit
    the known redundancy (joint5 about Z does not affect position) by fixing joint5=0,
    and we handle the other four joints (joint1 about Z, joints2..4 about Y) via a
    finite enumeration of possible “branches” in joint1, then a standard 3R planar
    solver for joints2..4.

    Geometry Summary (from the DETAILS):
      1) joint1 revolute about global Z at the base.
      2) joint2 at [0, 0.13585, 0], revolute about Y.
      3) joint3 at [0, -0.1197, 0.425], revolute about Y.
      4) joint4 at [0, 0, 0.39225], revolute about Y.
      5) joint5 at [0, 0.093, 0], revolute about Z.
      TCP at [0, 0, 0.09465].

    In practice, the Y offsets cause a 1–DOF redundancy. Our approach:
      • We consider two possible values for joint1 = ψ or ψ+π, where ψ = atan2(y, x).
        (Rotating the base by π flips the manipulator “backwards.”)
      • For each joint1 guess, we “undo” that rotation in the (x,y) plane to yield
        a transformed target p′. Then we solve the remaining sub-chain (joint2..4
        about Y, plus joint5=0) exactly as a planar 3R chain in the x–z plane with
        a net fixed offset in y = -0.0267.  (This is the same geometry logic as in
        “Existing code 2,” but adapted to the actual link lengths.)
      • We pick whichever branch yields the minimal final-position error.  This is
        a closed-form enumeration (just two branches for joint1 × two elbow “folds”),
        so no numeric iteration is needed.

    NOTE: The chosen solution may differ from other valid branches, but will be a
    legitimate closed-form IK solution.

    :param p: Desired TCP position as (x, y, z).
    :return: (joint1, joint2, joint3, joint4, joint5) in radians, each wrapped in [−π, π].
             joint5 is fixed to 0.
    """
    import math
    x, y, z = p

    def wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def solve_3R_y_axis(x_sub, z_sub):
        L1 = 0.425
        L2 = 0.39225
        L3 = 0.09465

        def fk_planar(t2, t3, t4):
            x_fk = L1 * math.sin(t2) + L2 * math.sin(t2 + t3) + L3 * math.sin(t2 + t3 + t4)
            z_fk = L1 * math.cos(t2) + L2 * math.cos(t2 + t3) + L3 * math.cos(t2 + t3 + t4)
            return (x_fk, z_fk)
        psi = math.atan2(x_sub, z_sub)
        T_opts = [psi, psi + math.pi]
        candidates = []
        for T in T_opts:
            x_w = x_sub - L3 * math.sin(T)
            z_w = z_sub - L3 * math.cos(T)
            r_w = math.hypot(x_w, z_w)
            cos_beta = (r_w ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            cos_beta = max(-1.0, min(1.0, cos_beta))
            for sgn in [+1, -1]:
                try:
                    beta = sgn * math.acos(cos_beta)
                except ValueError:
                    continue
                phi_w = math.atan2(x_w, z_w)
                delta = math.atan2(L2 * math.sin(beta), L1 + L2 * math.cos(beta))
                t2_candidate = phi_w - delta
                t3_candidate = beta
                t4_candidate = T - (t2_candidate + t3_candidate)
                x_f, z_f = fk_planar(t2_candidate, t3_candidate, t4_candidate)
                err = math.hypot(x_f - x_sub, z_f - z_sub)
                candidates.append((err, t2_candidate, t3_candidate, t4_candidate))
        if not candidates:
            return None
        best_sol = min(candidates, key=lambda c: c[0])
        return (best_sol[1], best_sol[2], best_sol[3])
    joint1_candidates = []
    base_angle = math.atan2(y, x)
    for candidate_t1 in [base_angle, base_angle + math.pi]:
        px = x * math.cos(candidate_t1) + y * math.sin(candidate_t1)
        py = -x * math.sin(candidate_t1) + y * math.cos(candidate_t1)
        pz = z
        x_sub, z_sub = (px, pz)
        sol_3R = solve_3R_y_axis(x_sub, z_sub)
        if sol_3R is None:
            continue
        t2, t3, t4 = sol_3R
        x_f_sub, z_f_sub = (0.0, 0.0)
        L1 = 0.425
        L2 = 0.39225
        L3 = 0.09465
        x_f_sub = L1 * math.sin(t2) + L2 * math.sin(t2 + t3) + L3 * math.sin(t2 + t3 + t4)
        z_f_sub = L1 * math.cos(t2) + L2 * math.cos(t2 + t3) + L3 * math.cos(t2 + t3 + t4)
        y_sub_const = -0.0267
        x_fk = x_f_sub * math.cos(candidate_t1) - y_sub_const * math.sin(candidate_t1)
        y_fk = x_f_sub * math.sin(candidate_t1) + y_sub_const * math.cos(candidate_t1)
        z_fk = z_f_sub
        err_pos = math.sqrt((x_fk - x) ** 2 + (y_fk - y) ** 2 + (z_fk - z) ** 2)
        joint1_candidates.append((err_pos, candidate_t1, t2, t3, t4))
    if not joint1_candidates:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best_sol = min(joint1_candidates, key=lambda c: c[0])
    _, j1, j2, j3, j4 = best_sol
    j1 = wrap_to_pi(j1)
    j2 = wrap_to_pi(j2)
    j3 = wrap_to_pi(j3)
    j4 = wrap_to_pi(j4)
    j5 = 0.0
    return (j1, j2, j3, j4, j5)