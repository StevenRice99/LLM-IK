import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Analytical closed–form inverse kinematics for a 5-DOF serial manipulator.
    
    Robot chain (all lengths in meters) and fixed translations come from:
      • Joint1 (revolute about Y) at origin.
      • Translation to Joint2: [0, -0.1197, 0.425]
      • Joint2: revolute about Y.
      • Translation to Joint3: [0, 0, 0.39225]
      • Joint3: revolute about Y.
      • Translation to Joint4: [0, 0.093, 0]
      • Joint4: revolute about Z.
      • Translation to Joint5: [0, 0, 0.09465]
      • Joint5: revolute about Y.
      • Translation to TCP: [0, 0.0823, 0]
    
    In this formulation, only joints 1-3 and joint4 affect the TCP position.
    In particular, when one derives the forward kinematics the TCP coordinates (x, y, z)
    come out as follows:
      S = q1 + q2 + q3   (sum of the three Y–axis rotations)
      x = L1 * sin(q1) + L2 * sin(q1+q2) + L3 * sin(S) - 0.0823 * sin(q4) * cos(S)
      z = L1 * cos(q1) + L2 * cos(q1+q2) + L3 * cos(S) + 0.0823 * sin(q4) * sin(S)
      y = -0.1197 + 0 + 0.093 + 0.0823*cos(q4)
        = -0.0267 + 0.0823*cos(q4)
    where
      L1 = 0.425      (translation along z in joint2)
      L2 = 0.39225    (translation along z in joint3)
      L3 = 0.09465    (translation along z from joint5 to TCP)
      
    The key observation is that the vertical (y) coordinate depends only on q4:
         y = -0.0267 + 0.0823*cos(q4)
    so we can solve for q4 immediately from the target y.
    Then, substituting d = 0.0823*sin(q4), one can show that the x and z equations become:
         x = P + L_eff * sin(T)
         z = Q + L_eff * cos(T)
    where
         P = L1*sin(q1) + L2*sin(q1+q2)
         Q = L1*cos(q1) + L2*cos(q1+q2)
         L_eff = sqrt(L3**2 + d**2)
         φ = atan2(d, L3)
         T = S - φ   with S = q1+q2+q3.
    A natural candidate for T is obtained from the target horizontal direction.
    Let ψ = atan2(x, z). Then one may set a candidate T = ψ (or ψ + π),
    and hence S = T + φ.
    Finally, the remaining 2R-subchain (joints 1 and 2) must produce a wrist center W,
    defined by:
         W = [ x - L_eff*sin(T),  z - L_eff*cos(T) ]
    The 2R IK yields q1 and q2 (with two solutions) and then q3 = S – (q1+q2).
    Joint5 is redundant for position, so we set it to zero.
    
    This function searches over the candidate branches for q4 (the two solutions
    from the y–equation) and for T (using ψ and ψ+π) and for the two possible
    2R IK solutions, then returns the candidate whose forward kinematics best
    matches the target.
    
    :param p: The target TCP position as (x, y, z).
              Note: p must be reachable. (y is not forced to a single value.)
    :return: A tuple (q1, q2, q3, q4, q5) in radians.
    """
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    y_offset = -0.1197 + 0.093
    tcp_y_offset = 0.0823
    x_target, y_target, z_target = p

    def normalize(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def fk(q1, q2, q3, q4):
        S = q1 + q2 + q3
        d = 0.0823 * math.sin(q4)
        x_fk = L1 * math.sin(q1) + L2 * math.sin(q1 + q2) + L3 * math.sin(S) - d * math.cos(S)
        z_fk = L1 * math.cos(q1) + L2 * math.cos(q1 + q2) + L3 * math.cos(S) + d * math.sin(S)
        y_fk = y_offset + tcp_y_offset * math.cos(q4)
        return (x_fk, y_fk, z_fk)
    C = (y_target - y_offset) / tcp_y_offset
    C = max(min(C, 1.0), -1.0)
    q4_candidates = [math.acos(C), -math.acos(C)]
    psi = math.atan2(x_target, z_target)
    best_error = float('inf')
    best_solution = None
    for q4_candidate in q4_candidates:
        d = tcp_y_offset * math.sin(q4_candidate)
        L_eff = math.sqrt(L3 ** 2 + d ** 2)
        phi = math.atan2(d, L3)
        for T_candidate in [psi, psi + math.pi]:
            S = T_candidate + phi
            W_x = x_target - L_eff * math.sin(T_candidate)
            W_z = z_target - L_eff * math.cos(T_candidate)
            r_w = math.hypot(W_x, W_z)
            if r_w > L1 + L2 or r_w < abs(L1 - L2):
                continue
            cos_q2 = (r_w ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            cos_q2 = max(min(cos_q2, 1.0), -1.0)
            for sign in [1, -1]:
                q2_candidate = sign * math.acos(cos_q2)
                delta = math.atan2(L2 * math.sin(q2_candidate), L1 + L2 * math.cos(q2_candidate))
                theta_w = math.atan2(W_x, W_z)
                q1_candidate = theta_w - delta
                q3_candidate = S - (q1_candidate + q2_candidate)
                x_fk, y_fk, z_fk = fk(q1_candidate, q2_candidate, q3_candidate, q4_candidate)
                error = math.sqrt((x_fk - x_target) ** 2 + (y_fk - y_target) ** 2 + (z_fk - z_target) ** 2)
                if error < best_error:
                    best_error = error
                    best_solution = (q1_candidate, q2_candidate, q3_candidate, q4_candidate, 0.0)
    if best_solution is None:
        raise ValueError('No valid IK solution found for the input target position.')
    q1, q2, q3, q4, q5 = best_solution
    q1 = normalize(q1)
    q2 = normalize(q2)
    q3 = normalize(q3)
    q4 = normalize(q4)
    q5 = normalize(q5)
    return (q1, q2, q3, q4, q5)