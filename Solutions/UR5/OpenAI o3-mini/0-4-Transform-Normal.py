def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Computes a closed‐form analytical solution for the inverse kinematics of the 5-DOF serial manipulator.
    
    Kinematic summary:
      • Joint 1: Revolute about Z at [0,0,0].
      • Joint 2: Translate [0, 0.13585, 0] then rotate about Y.
      • Joint 3: Translate [0, -0.1197, 0.425] then rotate about Y.
      • Joint 4: Translate [0, 0, 0.39225] then rotate about Y.
      • Joint 5: Translate [0, 0.093, 0] then rotate about Z.
      • TCP: Translate [0, 0, 0.09465].
    
    In a factorized form the forward kinematics are:
         R_total = Rz(q1) · [Ry(q2+q3+q4) · Rz(q5)]
         p_TCP = Rz(q1) * { planar_pos + tcp_offset }.
    
    The fixed translations along Y from joints 2 and 5 (and the negative offset in joint 3)
      yield a constant:
         y_const = 0.13585 - 0.1197 + 0.093 = 0.10915.
    
    In the base XY–plane (before the planar 2R subchain) the rotated target p̄ = Rz(–q1)·p must satisfy:
         p̄_y = -p_x*sin(q1) + p_y*cos(q1) = y_const.
    This equation has two solutions. We select the proper q1 branch by “testing” the decoupled
    orientation (see below).
    
    The desired orientation R_des is built from the provided roll–pitch–yaw (r) using
         R_des = Rz(yaw) · Ry(pitch) · Rx(roll)
    (the typical URDF convention), and then decoupled via
         M = Rz(–q1) · R_des.
    With the structure of the kinematics, one may show that M = Ry(φ)·Rz(q5) where
         φ = q2+q3+q4,
         q5 = atan2( M[1,0], M[1,1] ),
         and φ = atan2( M[0,2], M[2,2] ).

    The effective (planar) 2R arm is then obtained by “removing” the TCP offset along the direction φ.
    If we define:
         L1 = 0.425,    L2 = 0.39225,   L_tcp = 0.09465,
         and compute:
           P_x = p̄_x – L_tcp*sin(φ)
           P_z = p̄_z – L_tcp*cos(φ)
         then the 2R geometry yields (with r2 = √(P_x²+P_z²)):
             cos(q3) = (r2² – L1² – L2²)/(2*L1*L2)
         (with two branches: q3 = ±acos(…)).
         Then,
             q2 = atan2(P_x, P_z) – atan2( L2*sin(q3), L1 + L2*cos(q3) )
         and finally,
             q4 = φ – (q2 + q3).

    This implementation uses branch–selection (by “testing” the reconstructed positions)
    both for q1 (using the extra constraint hidden in the decoupled orientation matrix)
    and for the 2R solution for q3.
    """
    import math
    import numpy as np
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    p_x, p_y, p_z = p
    r_xy = math.sqrt(p_x ** 2 + p_y ** 2)
    theta = math.atan2(p_y, p_x)
    ratio = y_const / r_xy
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_candidate1 = theta - a
    q1_candidate2 = theta - (math.pi - a)

    def compute_M(q1_val):
        cos_q1 = math.cos(q1_val)
        sin_q1 = math.sin(q1_val)
        Rz_neg_q1 = np.array([[cos_q1, sin_q1, 0], [-sin_q1, cos_q1, 0], [0, 0, 1]])
        roll, pitch, yaw = r
        R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        R_des = R_z @ R_y @ R_x
        M_val = Rz_neg_q1 @ R_des
        return M_val
    M1 = compute_M(q1_candidate1)
    M2 = compute_M(q1_candidate2)
    err1 = abs(M1[1, 2])
    err2 = abs(M2[1, 2])
    q1 = q1_candidate1 if err1 <= err2 else q1_candidate2
    cos_q1 = math.cos(q1)
    sin_q1 = math.sin(q1)
    Rz_neg_q1 = np.array([[cos_q1, sin_q1, 0], [-sin_q1, cos_q1, 0], [0, 0, 1]])
    p_vec = np.array([p_x, p_y, p_z])
    p_bar = Rz_neg_q1 @ p_vec
    p_bar_x, p_bar_y, p_bar_z = p_bar
    roll, pitch, yaw = r
    R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_des = R_z @ R_y @ R_x
    M = Rz_neg_q1 @ R_des
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = math.atan2(M[1, 0], M[1, 1])
    P_x = p_bar_x - L_tcp * math.sin(phi)
    P_z = p_bar_z - L_tcp * math.cos(phi)
    r2 = math.sqrt(P_x ** 2 + P_z ** 2)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_candidateA = math.acos(cos_q3)
    q3_candidateB = -q3_candidateA

    def planar_solution(q3_val):
        q2_val = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q4_val = phi - (q2_val + q3_val)
        calc_x = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L_tcp * math.sin(phi)
        calc_z = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L_tcp * math.cos(phi)
        err_val = math.sqrt((calc_x - p_bar_x) ** 2 + (calc_z - p_bar_z) ** 2)
        return (q2_val, q4_val, err_val)
    q2_A, q4_A, err_A = planar_solution(q3_candidateA)
    q2_B, q4_B, err_B = planar_solution(q3_candidateB)
    if err_A <= err_B:
        q3 = q3_candidateA
        q2 = q2_A
        q4 = q4_A
    else:
        q3 = q3_candidateB
        q2 = q2_B
        q4 = q4_B
    return (q1, q2, q3, q4, q5)