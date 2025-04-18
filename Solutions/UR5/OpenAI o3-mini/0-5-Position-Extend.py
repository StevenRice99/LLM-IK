import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Analytical inverse kinematics for the 6‑DOF serial manipulator.
    
    Manipulator geometry (all lengths in meters, angles in radians):
      • Joint 1 (revolute): origin [0, 0, 0], axis Z.
      • Joint 2 (revolute): origin [0, 0.13585, 0], axis Y.
      • Joint 3 (revolute): origin [0, -0.1197, 0.425], axis Y.
      • Joint 4 (revolute): origin [0, 0, 0.39225], axis Y.
      • Joint 5 (revolute): origin [0, 0.093, 0], axis Z.
      • Joint 6 (revolute): origin [0, 0, 0.09465], axis Y.
      • TCP: position [0, 0.0823, 0] relative to Joint 6.
      
    The forward kinematics is built as a chain:
      Base → (Joint1) → translation by [0, 0.13585, 0] →
           (Joint2 about Y) → translation by [0, -0.1197, 0.425] →
           (Joint3 about Y) → translation by [0, 0, 0.39225] →
           (Joint4 about Y) → [Wrist subchain from Joint4]
           Wrist subchain:
              (Joint5 about Z): translation by [0, 0.093, 0]
              (Joint6 about Y): translation by [0, 0, 0.09465]
              TCP offset: [0, 0.0823, 0]
              
    In our method the inverse kinematics is solved by “decoupling” the arm
    into an arm (joints 1–4) and a wrist (joints 5–6). For our purposes we use a grid‐search
    over candidate wrist (Joint5) angles (setting Joint6 = 0) and then solve a 5‑DOF
    sub–problem (similar to an existing solution) for joints 1–4. In the 5‑DOF sub–chain,
    the procedure forces the rotated target (for an “arm” target point p_arm) to have a fixed y–coordinate
    in the frame of Joint2; the constant is chosen as:
          y_chain = (–0.1197 + 0.093) = –0.0267,
    reflecting the geometry from Joint2 to Joint3.
    
    Because the wrist adds an offset (dependent on q5) the arm target is first “backed out”
    from the TCP target p by subtracting the wrist offset. In our model the wrist offset,
    expressed in the frame of Joint4, is:
          p_offset = [ -a5*sin(q5),
                        a5*cos(q5) + tcp_y_offset,
                        L3 ]
    where
          a5 = 0.093,
          tcp_y_offset = 0.0823,
          L3 = 0.09465.
    Thus, the desired arm (or “wrist‐center”) target is:
          p_arm = p – p_offset.
    
    Then, using the familiar “EXISTING” 5‑DOF solution method (which forces the rotated target,
    into Joint2’s frame, to have y = y_chain), we compute candidate solutions for joints 1–4.
    
    Finally, we choose the candidate (over a grid of wrist angles q5) that minimizes the error
    in reaching the overall TCP position p. Joint6 is set to 0.
    
    Note:
      This solution returns one valid branch. Owing to multiple IK solutions, joint angles
      differing by integer multiples of 2π (or with alternate branch selection) are equivalent.
    
    :param p: Desired TCP position [x, y, z].
    :return: A 6‑tuple (q1, q2, q3, q4, q5, q6) in radians.
    """
    d2_y = 0.13585
    y_chain_target = -0.1197 + 0.093
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    a5 = 0.093
    tcp_y_offset = 0.0823
    best_solution = None
    best_err = float('inf')
    q5_candidates = np.linspace(-math.pi, math.pi, 20, endpoint=False)
    for q5 in q5_candidates:
        offset_x = -a5 * math.sin(q5)
        offset_y = a5 * math.cos(q5) + tcp_y_offset
        offset_z = L3
        p_arm_x = p[0] - offset_x
        p_arm_y = p[1] - offset_y
        p_arm_z = p[2] - offset_z
        delta = y_chain_target + d2_y
        x_w = p_arm_x
        y_w = p_arm_y
        z_w = p_arm_z
        r_xy = math.hypot(x_w, y_w)
        if r_xy < 1e-09:
            q1_list = [0.0]
        else:
            phi = math.atan2(-x_w, y_w)
            arg = delta / r_xy
            arg = max(-1.0, min(1.0, arg))
            gamma = math.acos(arg)
            q1_list = [phi + gamma, phi - gamma]
        for q1 in q1_list:
            c1 = math.cos(q1)
            s1 = math.sin(q1)
            x2 = c1 * x_w + s1 * y_w
            y2 = -s1 * x_w + c1 * y_w - d2_y
            z2 = z_w
            psi = math.atan2(x2, z2)
            for T in [psi, psi + math.pi]:
                xw = x2 - L3 * math.sin(T)
                zw = z2 - L3 * math.cos(T)
                rw2 = xw * xw + zw * zw
                cos_q3 = (rw2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
                cos_q3 = max(-1.0, min(1.0, cos_q3))
                for sign in [1.0, -1.0]:
                    try:
                        q3 = sign * math.acos(cos_q3)
                    except ValueError:
                        continue
                    phi_w = math.atan2(xw, zw)
                    delta_w = math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
                    q2 = phi_w - delta_w
                    q4 = T - (q2 + q3)
                    x_fk = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) + L3 * math.sin(q2 + q3 + q4)
                    z_fk = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) + L3 * math.cos(q2 + q3 + q4)
                    err_sub = (x_fk - x2) ** 2 + (z_fk - z2) ** 2 + (y2 - y_chain_target) ** 2
                    if err_sub < 1e-06:
                        p_arm_candidate_x = c1 * x2 - s1 * (y_chain_target + d2_y)
                        p_arm_candidate_y = s1 * x2 + c1 * (y_chain_target + d2_y)
                        p_arm_candidate_z = z2
                        p_fk_x = p_arm_candidate_x + offset_x
                        p_fk_y = p_arm_candidate_y + offset_y
                        p_fk_z = p_arm_candidate_z + offset_z
                        err_full = (p_fk_x - p[0]) ** 2 + (p_fk_y - p[1]) ** 2 + (p_fk_z - p[2]) ** 2
                        if err_full < best_err:
                            best_err = err_full
                            best_solution = (q1, q2, q3, q4, q5, 0.0)
    if best_solution is None:
        raise ValueError('No IK solution found for the target position.')

    def normalize(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    q_sol = tuple((normalize(a) for a in best_solution))
    return q_sol