def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A direct, closed-form approach for a 5-DOF serial manipulator with joint layout:
      • q1, q2, q3 all revolve about Y (shoulder, elbow, etc.)
      • q4 revolves about Z
      • q5 revolves about Y
      • Then a final fixed transform to the TCP (translate + rotate Z by +π/2)

    Because the robot has only 5 DOFs but we want position + orientation, we cannot
    generally match all 3 orientation angles independently (typical for 6-DOF).
    However, the problem states all given targets are in the achievable set.

    This method:
     1) Extracts a "wrist orientation" for q4, q5 from the desired orientation (r),
        matching (roughly) the final two rotation axes (Z first, then Y),
        ignoring any leftover roll about the TCP's X axis that the robot cannot realize.
     2) Once q4, q5 are chosen, compute the effective wrist offset in the manipulator's
        frame and subtract it from the desired position p to find the 3-DOF portion that
        q1, q2, q3 must reach.
     3) Solve for q1, q2, q3 via standard geometry in the plane, since all three revolve about Y.
     4) Wrap final angles into [-π, +π], then clamp to the allowed range [-6.2831853, +6.2831853].
    No loops, no numeric iteration. Pure direct trig should be fast.

    Returns (q1, q2, q3, q4, q5) in radians, each within [-6.2831853, 6.2831853].
    """
    import math
    px, py, pz = p
    rx, ry, rz = r
    dY_12 = -0.1197
    dZ_12 = 0.425
    dZ_23 = 0.39225
    dY_34 = 0.093
    dZ_45 = 0.09465
    dY_TCP = 0.0823
    q4 = rz - math.pi / 2.0
    q5 = ry
    import math

    def rotZ(a):
        c = math.cos(a)
        s = math.sin(a)
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    def rotY(a):
        c = math.cos(a)
        s = math.sin(a)
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

    def mat_vec3(mat, vec):
        return [mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2], mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2], mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2]]
    link4_offset = [0, 0.093, 0]
    Rz4 = rotZ(q4)
    link5_offset_in_link4 = mat_vec3(Rz4, [0, 0, 0.09465])
    partial_4_5 = [link4_offset[0] + link5_offset_in_link4[0], link4_offset[1] + link5_offset_in_link4[1], link4_offset[2] + link5_offset_in_link4[2]]
    Ry5 = rotY(q5)
    Rz4Ry5 = [[Rz4[0][0] * Ry5[0][0] + Rz4[0][1] * Ry5[1][0] + Rz4[0][2] * Ry5[2][0], Rz4[0][0] * Ry5[0][1] + Rz4[0][1] * Ry5[1][1] + Rz4[0][2] * Ry5[2][1], Rz4[0][0] * Ry5[0][2] + Rz4[0][1] * Ry5[1][2] + Rz4[0][2] * Ry5[2][2]], [Rz4[1][0] * Ry5[0][0] + Rz4[1][1] * Ry5[1][0] + Rz4[1][2] * Ry5[2][0], Rz4[1][0] * Ry5[0][1] + Rz4[1][1] * Ry5[1][1] + Rz4[1][2] * Ry5[2][1], Rz4[1][0] * Ry5[0][2] + Rz4[1][1] * Ry5[1][2] + Rz4[1][2] * Ry5[2][2]], [Rz4[2][0] * Ry5[0][0] + Rz4[2][1] * Ry5[1][0] + Rz4[2][2] * Ry5[2][0], Rz4[2][0] * Ry5[0][1] + Rz4[2][1] * Ry5[1][1] + Rz4[2][2] * Ry5[2][1], Rz4[2][0] * Ry5[0][2] + Rz4[2][1] * Ry5[1][2] + Rz4[2][2] * Ry5[2][2]]]
    tcp_in_link5 = [0, 0, 0.0823]
    tcp_in_link4 = mat_vec3(Rz4Ry5, tcp_in_link5)
    partial_4_5_tcp = [partial_4_5[0] + tcp_in_link4[0], partial_4_5[1] + tcp_in_link4[1], partial_4_5[2] + tcp_in_link4[2]]
    wrist_offset_approx = math.sqrt(partial_4_5_tcp[0] * partial_4_5_tcp[0] + partial_4_5_tcp[1] * partial_4_5_tcp[1] + partial_4_5_tcp[2] * partial_4_5_tcp[2])
    q1 = math.atan2(px, pz)
    base_r = math.sqrt(px * px + pz * pz)
    r_eff = base_r - wrist_offset_approx
    if r_eff < 0:
        r_eff = 0
    L1 = 0.425
    L2 = 0.39225
    py_eff = py + 0.1197
    plane_d = math.sqrt(r_eff * r_eff + py_eff * py_eff)
    if plane_d < 1e-06:
        q2 = 0.0
        q3 = 0.0
    else:
        cos_q3 = (L1 * L1 + L2 * L2 - plane_d * plane_d) / (2 * L1 * L2)
        if cos_q3 > 1.0:
            cos_q3 = 1.0
        if cos_q3 < -1.0:
            cos_q3 = -1.0
        elbow = math.acos(cos_q3)
        q3plane = math.pi - elbow
        alpha = math.atan2(py_eff, r_eff)
        sin_q3plane = math.sin(q3plane)
        cos_q3plane = math.cos(q3plane)
        denom = L1 + L2 * cos_q3plane
        if abs(denom) < 1e-09:
            q2plane = alpha
        else:
            beta = math.atan2(L2 * sin_q3plane, denom)
            q2plane = alpha - beta
        q2 = q2plane
        q3 = q3plane

    def wrap_clamp(a):
        from math import pi
        while a > pi:
            a -= 2 * pi
        while a < -pi:
            a += 2 * pi
        if a > 6.2831853:
            a = 6.2831853
        if a < -6.2831853:
            a = -6.2831853
        return float(a)
    q1 = wrap_clamp(q1)
    q2 = wrap_clamp(q2)
    q3 = wrap_clamp(q3)
    q4 = wrap_clamp(q4)
    q5 = wrap_clamp(q5)
    return (q1, q2, q3, q4, q5)