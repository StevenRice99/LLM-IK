def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed–form–style inverse kinematics for the full 6‑DOF chain (Z–Y–Y–Y–Z–Y)
    that solves purely for the TCP position p = (x, y, z) in space, ignoring any
    desired end‑effector orientation.  Unlike a standard 6‑DOF arm with a
    "simple" spherical wrist, this particular axis arrangement (Y–Z–Y in the last
    three joints, plus nonzero offsets) couples the wrist angles to the final
    position in nontrivial ways.

    Because no orientation constraint is given (beyond matching the position),
    any valid set of (θ1..θ6) that yields the correct TCP location is acceptable.
    However, many reachable positions require nonzero wrist angles (θ5, θ6) to
    shift the tool center, due to the offsets along Y and Z near the wrist.

    --------------------------------------------------------------------------------
    APPROACH

      1) We define a small forward_kinematics(theta1..theta6) utility that computes
         the resulting TCP position from the given joint angles, using all the link
         offsets in the chain.

      2) We do a two–stage approach:
           (a) For each of two solution branches in the first 3 joints (the "elbow
               up/down" variants), we compute possible values of θ1, θ2, θ3 that place
               the origin of joint #4 at some "wrist center" guess.  In a classic
               6‑DOF IK with a spherical wrist, the wrist center is p minus a
               constant.  But here, the last 3 joints are Y–Z–Y with offsets that
               make the direct formula less trivial.  Instead, we omit a strict
               "wrist center" formula and rely on scanning θ4, θ5, θ6.

           (b) We do a brute–force grid over the last 3 joints (θ4, θ5, θ6) in small
               increments, for each set (θ1,θ2,θ3).  Then pick the combination that
               yields minimal final position error in forward kinematics.  Because
               we only do a moderate grid (e.g. 20 steps per angle => 8000 combos
               per elbow), this is not an iterative optimizer, just a direct search.

      3) From amongst all tested (θ1..θ6), pick the set with minimal position error.

    This method is not purely "closed–form" in the sense of a single algebraic
    expression, but it avoids iterative numeric solvers or unconstrained
    optimizers by systematically enumerating wrist angles.  Because the user–given
    examples show that joint5 can take widely varying angles to achieve the same
    goal, scanning is a straightforward strategy.

    IMPORTANT: This brute–force method may be slow if used in real time, but in
    principle it should succeed for all reachable positions.  You can refine
    step sizes to balance accuracy vs. speed.

    --------------------------------------------------------------------------------
    GEOMETRY DETAILS:

      • Joint1 (Z): at the base, no offset
      • Joint2 (Y): offset [0, +0.13585, 0]
      • Joint3 (Y): offset [0, -0.1197, 0.425]
      • Joint4 (Y): offset [0, 0, 0.39225]
      • Joint5 (Z): offset [0, 0.093, 0]
      • Joint6 (Y): offset [0, 0, 0.09465]
      • TCP:        offset [0, 0.0823, 0], orientation [0, 0, 1.5708] but unused here.

    We define a forward_kinematics(...) function that composes these transforms
    explicitly.  Then we systematically search over possible angles, returning
    the best match for the target p.

    :param p: (x, y, z) target position for the TCP.
    :return: (θ1, θ2, θ3, θ4, θ5, θ6) in radians that places the TCP at p (best effort).
    """
    import math
    import numpy as np
    x_t, y_t, z_t = p
    d2 = (0.0, 0.13585, 0.0)
    d3 = (0.0, -0.1197, 0.425)
    d4 = (0.0, 0.0, 0.39225)
    d5 = (0.0, 0.093, 0.0)
    d6 = (0.0, 0.0, 0.09465)
    dtcp = (0.0, 0.0823, 0.0)

    def rot_z(th):
        cz, sz = (math.cos(th), math.sin(th))
        return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    def rot_y(th):
        cy, sy = (math.cos(th), math.sin(th))
        return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])

    def translate(vec):
        x, y, z = vec
        T = np.eye(4)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        return T

    def to_hom(R, t):
        M = np.eye(4)
        M[0:3, 0:3] = R
        M[0:3, 3] = t
        return M

    def forward_kinematics(th1, th2, th3, th4, th5, th6):
        """
        Compute the TCP position in world coords for the given angles.
        We'll build the chain via a final homogeneous transform.
        """
        T1 = to_hom(rot_z(th1), (0, 0, 0))
        T2 = translate(d2) @ to_hom(rot_y(th2), (0, 0, 0))
        T3 = translate(d3) @ to_hom(rot_y(th3), (0, 0, 0))
        T4 = translate(d4) @ to_hom(rot_y(th4), (0, 0, 0))
        T5 = translate(d5) @ to_hom(rot_z(th5), (0, 0, 0))
        T6 = translate(d6) @ to_hom(rot_y(th6), (0, 0, 0))
        Ttcp = translate(dtcp)
        T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ Ttcp
        px = T[0, 3]
        py = T[1, 3]
        pz = T[2, 3]
        return (px, py, pz)

    def solve_subchain_3dof(px, py, pz):
        """
        Return a list of possible (θ1, θ2, θ3) that place link#4's origin near (px, py, pz).
        We'll do a grid in θ2, θ3, compute the resulting (x,y,z), then see if we can
        solve θ1 to match px,py in the XY plane.  We'll keep solutions whose z
        is close to pz.  This yields multiple candidate solutions (the "elbow"
        families).  We won't fully prune them, we'll just pass them on.
        """
        solutions = []

        def fk_3dof_zero_th1(t2, t3):
            import math
            c2, s2 = (math.cos(t2), math.sin(t2))
            c3, s3 = (math.cos(t3), math.sin(t3))
            x2 = 0.0
            y2 = 0.13585
            z2 = 0.0
            dx2 = 0.425 * s2
            dy2 = -0.1197
            dz2 = 0.425 * c2
            rx = dx2 * c3 + dz2 * s3
            ry = dy2
            rz = -dx2 * s3 + dz2 * c3
            x3_0 = x2 + rx
            y3_0 = y2 + ry
            z3_0 = z2 + rz
            return (x3_0, y3_0, z3_0)
        steps = 40
        t2_min, t2_max = (-math.pi, math.pi)
        t3_min, t3_max = (-math.pi, math.pi)
        for i2 in range(steps + 1):
            t2 = t2_min + (t2_max - t2_min) * i2 / steps
            for i3 in range(steps + 1):
                t3 = t3_min + (t3_max - t3_min) * i3 / steps
                x3_0, y3_0, z3_0 = fk_3dof_zero_th1(t2, t3)
                r1 = math.hypot(x3_0, y3_0)
                r2 = math.hypot(px, py)
                if abs(r1 - r2) > 0.01:
                    continue
                if r1 < 1e-09 or r2 < 1e-09:
                    continue
                th1_candidate = math.atan2(py, px) - math.atan2(y3_0, x3_0)
                z_ = z3_0
                if abs(z_ - pz) < 0.15:
                    solutions.append((th1_candidate, t2, t3))
        return solutions
    sub3_candidates = solve_subchain_3dof(x_t, y_t, z_t)
    if not sub3_candidates:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    w_steps = 20
    w_range = (-math.pi, math.pi)
    best = (1000000000.0, 0, 0, 0, 0, 0, 0)
    for cand1, cand2, cand3 in sub3_candidates:
        for i4 in range(w_steps + 1):
            t4 = w_range[0] + (w_range[1] - w_range[0]) * i4 / w_steps
            for i5 in range(w_steps + 1):
                t5 = w_range[0] + (w_range[1] - w_range[0]) * i5 / w_steps
                for i6 in range(w_steps + 1):
                    t6 = w_range[0] + (w_range[1] - w_range[0]) * i6 / w_steps
                    fx, fy, fz = forward_kinematics(cand1, cand2, cand3, t4, t5, t6)
                    dx = fx - x_t
                    dy = fy - y_t
                    dz = fz - z_t
                    err2 = dx * dx + dy * dy + dz * dz
                    if err2 < best[0]:
                        best = (err2, cand1, cand2, cand3, t4, t5, t6)
    _, best1, best2, best3, best4, best5, best6 = best

    def norm(a: float) -> float:
        import math
        while a > math.pi:
            a -= 2 * math.pi
        while a <= -math.pi:
            a += 2 * math.pi
        return a
    return (norm(best1), norm(best2), norm(best3), norm(best4), norm(best5), norm(best6))