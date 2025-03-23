def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Closed-form inverse kinematics for the 5-DOF manipulator to reach the position p = (x, y, z)
    and orientation r = (roll, pitch, yaw) in radians.
    
    NOTES / APPROACH:
      • The manipulator structure from the DETAILS is:
          • J1: revolute about Y @ base
          • J2: revolute about Y
          • J3: revolute about Y
          • J4: revolute about Z
          • J5: revolute about Y
          • TCP offset: +90° about Z (rpy=[0,0,1.570796325]) from link5
      • The existing position-only solution lumps joints 1..3 about Y plus joint4 about Z
        to find x,y,z. Joint5 about Y only affects orientation, not position. 
      • Here, we extend that approach to also match orientation by systematically enumerating
        candidate solutions, including ±π branches for the orientation angles that can appear.
      • We clamp final angles to [-2π, 2π].  No additional checks for out-of-limits are done
        beyond that. 
      • Some targets may have multiple valid solutions. We pick the one that best approximates
        the requested pose by minimal squared error in position+orientation. 
      • This code attempts thorough sampling of different ±π branches for the final orientation,
        but certain specialized configurations can still be tricky.  If further improvements
        are needed, a more exhaustive search over angles or additional geometric constraints
        may be introduced.

    Returns (q1, q2, q3, q4, q5) in radians, each in the range [-2π, 2π].
    """
    import math
    import numpy as np
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    base_offset = -0.1197
    link4_offset_y = 0.093
    y_offset = base_offset + link4_offset_y
    tcp_y_offset = 0.0823

    def clamp_2pi(a: float) -> float:
        """Clamp angle a to the range [-2π, 2π]."""
        TWO_PI = 2.0 * math.pi
        if a > TWO_PI:
            a = TWO_PI
        elif a < -TWO_PI:
            a = -TWO_PI
        return a

    def normalize_pi(a: float) -> float:
        """Normalize angle a to (-π, π]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a <= -math.pi:
            a += 2.0 * math.pi
        return a

    def rot_x(rx: float) -> np.ndarray:
        c = math.cos(rx)
        s = math.sin(rx)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def rot_y(ry: float) -> np.ndarray:
        c = math.cos(ry)
        s = math.sin(ry)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def rot_z(rz: float) -> np.ndarray:
        c = math.cos(rz)
        s = math.sin(rz)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

    def orientation_error_sq(Rtest: np.ndarray, Rtarget: np.ndarray) -> float:
        diff = Rtest - Rtarget
        return float(np.sum(diff * diff))
    roll, pitch, yaw = r
    R_des = rot_x(roll) @ rot_y(pitch) @ rot_z(yaw)
    R_offset = rot_z(-math.pi / 2)
    R_prime = R_des @ R_offset

    def fk_position(q1, q2, q3, q4):
        S = q1 + q2 + q3
        d = tcp_y_offset * math.sin(q4)
        x_fk = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(S) - d * math.cos(S)
        z_fk = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(S) + d * math.sin(S)
        y_fk = y_offset + tcp_y_offset * math.cos(q4)
        return (x_fk, y_fk, z_fk)

    def fk_orientation_error(q1, q2, q3, q4, q5):
        S = q1 + q2 + q3
        R_S = rot_y(S)
        R_4 = rot_z(q4)
        R_5 = rot_y(q5)
        R_test = R_S @ R_4 @ R_5 @ rot_z(math.pi / 2)
        return orientation_error_sq(R_test, R_des)

    def solve_q5_candidates(S, q4):
        R_temp_inv = rot_z(-q4) @ rot_y(-S)
        R_double = R_temp_inv @ R_prime
        base = math.atan2(R_double[0, 2], R_double[0, 0])
        return [base, base + math.pi, base - math.pi]
    x_t, y_t, z_t = p
    denom = tcp_y_offset
    cos4_ = (y_t - y_offset) / denom
    cos4_ = max(-1.0, min(1.0, cos4_))
    try:
        q4_candidates = [math.acos(cos4_), -math.acos(cos4_)]
    except ValueError:
        raise ValueError('No valid solution for q4 based on y coordinate.')
    psi = math.atan2(x_t, z_t)
    T_list = [psi, psi + math.pi]
    best_err = float('inf')
    best_sol = None
    SHIFT_5 = [0.0, 2.0 * math.pi, -2.0 * math.pi]
    for q4_candidate in q4_candidates:
        d = tcp_y_offset * math.sin(q4_candidate)
        L_eff = math.sqrt(L3 ** 2 + d ** 2)
        phi = math.atan2(d, L3)
        for T_candidate in T_list:
            S_val = T_candidate + phi
            W_x = x_t - L_eff * math.sin(T_candidate)
            W_z = z_t - L_eff * math.cos(T_candidate)
            r_w = math.hypot(W_x, W_z)
            if r_w > L1 + L2 or r_w < abs(L1 - L2):
                continue
            c2_ = (r_w ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            c2_ = max(-1.0, min(1.0, c2_))
            for sgn in [1, -1]:
                try:
                    q2_0 = sgn * math.acos(c2_)
                except ValueError:
                    continue
                delta = math.atan2(L2 * math.sin(q2_0), L1 + L2 * math.cos(q2_0))
                th_w = math.atan2(W_x, W_z)
                q1_0 = th_w - delta
                q3_0 = S_val - (q1_0 + q2_0)
                x_fk, y_fk, z_fk = fk_position(q1_0, q2_0, q3_0, q4_candidate)
                pos_err = math.sqrt((x_fk - x_t) ** 2 + (y_fk - y_t) ** 2 + (z_fk - z_t) ** 2)
                base_q5_cands = solve_q5_candidates(q1_0 + q2_0 + q3_0, q4_candidate)
                for raw_q5 in base_q5_cands:
                    for shift5 in SHIFT_5:
                        q5_test = raw_q5 + shift5

                        def clamp_joints(a1, a2, a3, a4, a5):
                            return (clamp_2pi(a1), clamp_2pi(a2), clamp_2pi(a3), clamp_2pi(a4), clamp_2pi(a5))
                        q1cl, q2cl, q3cl, q4cl, q5cl = clamp_joints(q1_0, q2_0, q3_0, q4_candidate, q5_test)
                        ori_err = fk_orientation_error(q1cl, q2cl, q3cl, q4cl, q5cl)
                        x_fk2, y_fk2, z_fk2 = fk_position(q1cl, q2cl, q3cl, q4cl)
                        pos_err2 = math.sqrt((x_fk2 - x_t) ** 2 + (y_fk2 - y_t) ** 2 + (z_fk2 - z_t) ** 2)
                        total_err = pos_err2 + ori_err
                        if total_err < best_err:
                            best_err = total_err
                            best_sol = (q1cl, q2cl, q3cl, q4cl, q5cl)
    if best_sol is None:
        raise ValueError('No valid IK solution found for the given target pose.')
    q1f, q2f, q3f, q4f, q5f = [normalize_pi(a) for a in best_sol]
    return (q1f, q2f, q3f, q4f, q5f)