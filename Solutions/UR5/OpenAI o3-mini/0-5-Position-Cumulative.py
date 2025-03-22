import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Analytical closed–form inverse kinematics (position-only) for a 6–DOF manipulator.
    
    The robot is described by:
      • Joint 1: Revolute about Z at [0,0,0]
      • Joint 2: Revolute about Y located at translation [0, 0.13585, 0]
      • Joint 3: Revolute about Y at a relative translation [0, -0.1197, 0.425]
      • Joint 4: Revolute about Y at a relative translation [0, 0, 0.39225]
      • Joint 5: Revolute about Z at a relative translation [0, 0.093, 0]
      • Joint 6: Revolute about Y at a relative translation [0, 0, 0.09465]
      • TCP: with an offset [0, 0.0823, 0] in the final (joint 6) frame
      
    In the nominal configuration (all joint angles zero), the TCP is at:
          [0, 0.19145, 0.9119]
    The complete forward kinematics (FK) are obtained by sequentially “building up” the pose:
    
      T_total = T1(q1) • T2 • R_y(q2) • T3 • R_y(q3) • T4 • R_y(q4) • T5 • R_z(q5) • T6 • R_y(q6) • T_tcp
    
    where the fixed translations (in meters) are:
      T2: [0, 0.13585, 0]
      T3: [0, -0.1197, 0.425]
      T4: [0, 0, 0.39225]
      T5: [0, 0.093, 0]
      T6: [0, 0, 0.09465]
      T_tcp: [0, 0.0823, 0]
    
    The joint rotation axes are as indicated (with joint 1 and 5 about Z, the others about Y).
    (Note that joint 6 is redundant for position; we set q6 = 0.)
    
    Because the kinematics are fully decoupled the solution permits a closed–form branch selection.
    One first “decouples” the base rotation (q1) so that the remaining arm chain (joints 2–4)
    acts in an effective vertical (YZ) plane, then one uses standard 2R inverse–kinematics for
    a sub–chain, and finally solves for a wrist rotation q5.
    
    This implementation computes two sets of candidate solutions (reflecting the cosine ambiguities)
    and selects the candidate whose forward kinematics (computed below) best match the target.
    
    Note: This solution assumes that the input p is reachable and does no explicit reachability check.
    
    :param p: Desired TCP position (x, y, z)
    :return: A tuple (q1, q2, q3, q4, q5, q6) of joint angles in radians.
    """
    A = 0.13585
    B = -0.1197
    C = 0.425
    D = 0.39225
    E = 0.093
    F = 0.09465
    G = 0.0823

    def rot_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    def rot_y(theta):
        return np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])

    def fk(j1, j2, j3, j4, j5, j6):
        R = np.eye(3)
        pos = np.zeros(3)
        R = rot_z(j1) @ R
        pos = pos + R @ np.array([0, A, 0])
        R = R @ rot_y(j2)
        pos = pos + R @ np.array([0, B, C])
        R = R @ rot_y(j3)
        pos = pos + R @ np.array([0, 0, D])
        R = R @ rot_y(j4)
        pos = pos + R @ np.array([0, E, 0])
        R = R @ rot_z(j5)
        pos = pos + R @ np.array([0, 0, F])
        R = R @ rot_y(j6)
        pos = pos + R @ np.array([0, G, 0])
        return pos
    q1_candidate = -math.atan2(p[0], p[1])
    L1 = C
    L2 = D
    candidates = []
    p_nom_arm = np.array([0, 0.10915, 0.9119])
    for q5_candidate in [1.0, -1.0]:
        d = G * math.sin(q5_candidate)
        L_eff = math.sqrt(F ** 2 + d ** 2)
        phi = math.atan2(d, F)
        Rz_inv = np.array([[math.cos(q1_candidate), math.sin(q1_candidate), 0], [-math.sin(q1_candidate), math.cos(q1_candidate), 0], [0, 0, 1]])
        p_arm = Rz_inv @ np.array(p)
        Y_target = p_arm[1]
        Z_target = p_arm[2]
        delta_candidate = math.atan2(Y_target - 0.10915, Z_target - 0.9119)
        T_candidate = delta_candidate + phi
        T_candidate_alt = T_candidate + math.pi
        for T in [T_candidate, T_candidate_alt]:
            R_val = L_eff
            r = math.hypot(L_eff * math.cos(T - phi), L_eff * math.sin(T - phi))
            cos_q3 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            if abs(cos_q3) > 1:
                continue
            for sign in [1, -1]:
                q3_candidate = sign * math.acos(cos_q3)
                delta_angle = math.atan2(L2 * math.sin(q3_candidate), L1 + L2 * math.cos(q3_candidate))
                q2_candidate = math.atan2(L_eff * math.sin(T - phi), L_eff * math.cos(T - phi)) - delta_angle
                q4_candidate = T - (q2_candidate + q3_candidate)
                candidate = (q1_candidate, q2_candidate, q3_candidate, q4_candidate, q5_candidate, 0.0)
                pos_fk = fk(*candidate)
                error = math.sqrt((pos_fk[0] - p[0]) ** 2 + (pos_fk[1] - p[1]) ** 2 + (pos_fk[2] - p[2]) ** 2)
                candidates.append((error, candidate))
    if not candidates:
        raise ValueError('No valid IK solution found for the target position.')
    best_candidate = min(candidates, key=lambda x: x[0])[1]

    def normalize(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    best_candidate = tuple((normalize(ang) for ang in best_candidate))
    return best_candidate