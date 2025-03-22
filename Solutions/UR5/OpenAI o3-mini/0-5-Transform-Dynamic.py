def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‐form analytical inverse kinematics solution for the 6-DOF serial manipulator.
    
    Kinematic summary (from URDF details):
      • Joint 1: Revolute about Z at [0,0,0]
      • Joint 2: Translate [0, 0.13585, 0] then rotate about Y
      • Joint 3: Translate [0, -0.1197, 0.425] then rotate about Y
      • Joint 4: Translate [0, 0, 0.39225] then rotate about Y
      • Joint 5: Translate [0, 0.093, 0] then rotate about Z
      • Joint 6: Translate [0, 0, 0.09465] then rotate about Y
      • TCP: (has a fixed orientation offset R_tcp = Rz(1.570796325); note its link‐position is [0, 0.0823, 0])
    
    This solution decouples the problem in two stages.
      1. Solve the 5-DOF “arm” (joints 1–5) so that the wrist (joint 6) reaches the proper position.
         The approach is based on:
             – A constant offset in the decoupled Y direction: 
                     y_const = 0.13585 - 0.1197 + 0.093  = 0.10915.
             – An equation in the base XY–plane: 
                     -p_x*sin(q1) + p_y*cos(q1) = y_const.
         Two candidate solutions for q1 are generated and the one that minimizes the mismatch in this equation is chosen.
      
      2. With q1–q5 determined (via a decoupled planar 2R subchain), the remaining wrist rotation q6 is recovered
         from the residual orientation:
             R_des = R_arm · ( Ry(q6) · R_tcp )
         where R_arm = Rz(q1) · Ry(q2+q3+q4) · Rz(q5) and R_tcp = Rz(1.570796325).
         Then, Ry(q6) = (R_armᵀ · R_des) · R_tcpᵀ and q6 = atan2( X[0,2], X[0,0] ).
    
    Parameters:
      p : (x, y, z) target position of the TCP.
      r : (roll, pitch, yaw) target orientation (in radians) of the TCP, using URDF convention:
          R_des = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    
    Returns:
      (q1, q2, q3, q4, q5, q6)  joint angles in radians.
    """
    import math
    import numpy as np

    def normalize_angle(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi
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
    cand1 = theta - a
    cand2 = theta - (math.pi - a)

    def candidate_error(q1_val: float) -> float:
        return abs(-p_x * math.sin(q1_val) + p_y * math.cos(q1_val) - y_const)
    err1 = candidate_error(cand1)
    err2 = candidate_error(cand2)
    q1 = cand1 if err1 <= err2 else cand2
    q1 = normalize_angle(q1)
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

    def planar_solution(q3_val: float) -> tuple[float, float, float]:
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
    q1 = normalize_angle(q1)
    q2 = normalize_angle(q2)
    q3 = normalize_angle(q3)
    q4 = normalize_angle(q4)
    q5 = normalize_angle(q5)
    R_z_q1 = np.array([[math.cos(q1), -math.sin(q1), 0], [math.sin(q1), math.cos(q1), 0], [0, 0, 1]])
    angle_y = q2 + q3 + q4
    R_y_total = np.array([[math.cos(angle_y), 0, math.sin(angle_y)], [0, 1, 0], [-math.sin(angle_y), 0, math.cos(angle_y)]])
    R_z_q5 = np.array([[math.cos(q5), -math.sin(q5), 0], [math.sin(q5), math.cos(q5), 0], [0, 0, 1]])
    R_arm = R_z_q1 @ R_y_total @ R_z_q5
    R_tcp = np.array([[math.cos(1.570796325), -math.sin(1.570796325), 0], [math.sin(1.570796325), math.cos(1.570796325), 0], [0, 0, 1]])
    X = R_arm.T @ R_des @ R_tcp.T
    q6 = math.atan2(X[0, 2], X[0, 0])
    q6 = normalize_angle(q6)
    return (q1, q2, q3, q4, q5, q6)