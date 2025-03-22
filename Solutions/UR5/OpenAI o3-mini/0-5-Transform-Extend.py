import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes a closed‐form analytical inverse kinematics solution for the 6-DOF robot.

    Robot summary (all units in meters and radians):
      • Joint 1: revolute about Z at [0, 0, 0].
      • Joint 2: translate [0, 0.13585, 0] then revolute about Y.
      • Joint 3: translate [0, -0.1197, 0.425] then revolute about Y.
      • Joint 4: translate [0, 0, 0.39225] then revolute about Y.
      • Joint 5: translate [0, 0.093, 0] then revolute about Z.
      • Joint 6: translate [0, 0, 0.09465] then revolute about Y.
      • TCP: translate [0, 0.0823, 0] with a fixed rotation Rz(1.570796325).

    In our approach the chain is decoupled into:
      [1] A 5-DOF “arm” solution (joints 1–5) that assumes the TCP lies at the tip of link 6.
          (Here the constant that arises from the fixed translations is 
             y_const = 0.13585 – 0.1197 + 0.093 = 0.10915.)
      [2] A “wrist” solution that closes the orientation loop by solving
          R_des = R_arm · [Rz(q5)*Ry(q6)*R_tcp_offset],
        where R_tcp_offset ≡ Rz(1.570796325).

    The method proceeds as follows:
      1. Solve for q1 using the target TCP position “p” and the planar constraint
         –p_x*sin(q1) + p_y*cos(q1) = y_const. (This yields two candidates; we pick the one with lower error.)
      2. Compute the decoupled orientation M = Rz(–q1)·R_des,
         where R_des is built from the given roll–pitch–yaw (using URDF’s Rz·Ry·Rx convention).
         Then let φ = atan2(M[0,2], M[2,2]).
      3. Rotate “p” by –q1 (i.e. compute p_bar = Rz(–q1)·p) and “remove” the wrist translation along the axis φ.
         (Here L_tcp = 0.09465, the joint-6 translation.)
      4. With the resulting planar position (P_x, P_z) and using link lengths L1 = 0.425 and L2 = 0.39225,
         solve a 2R geometry for q2 and q3 (yielding two branches) then set q4 = φ – (q2+q3).
         (We select the branch that minimizes the reconstruction error.)
      5. At this point an initial 5-DOF solution is available:
             (q1, q2, q3, q4, q5_prelim)
         with q5_prelim = atan2(M[1,0], M[1,1]).
      6. Compute the “arm rotation” R_arm = Rz(q1)·Ry(q2+q3+q4).
      7. To “close the loop” for the full orientation, note that the complete forward kinematics are:
             R_des = R_arm · [Rz(q5) · Ry(q6) · R_tcp_offset].
         Rearranging gives:
             A = R_armᵀ · R_des · (R_tcp_offset)ᵀ = Rz(q5) · Ry(q6).
         Using the standard parameterization for a product Rz(q5)·Ry(q6):
             A[0,0] = cos(q5)*cos(q6),   A[1,0] = sin(q5)*cos(q6),   A[2,0] = –sin(q6),
         we extract:
             q6 = atan2( – A[2,0], A[2,2] )
         and update q5 as
             q5 = atan2( A[1,0], A[0,0] ).
      8. Return the full 6-tuple (q1, q2, q3, q4, q5, q6).

    Note:
      • Branch–selection (for q1 and the planar 2R sub-solution) is based on minimizing reconstruction error.
      • There is an inherent redundancy in representing the wrist; the extraction of q5 and q6 is
        done in a way that yields the same overall rotation.

    :param p: Desired TCP position [x, y, z].
    :param r: Desired TCP orientation expressed as (roll, pitch, yaw) in radians.
    :return: A tuple (q1, q2, q3, q4, q5, q6) of joint angles in radians.
    """
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    angle_tcp = 1.570796325
    R_tcp_offset = np.array([[math.cos(angle_tcp), -math.sin(angle_tcp), 0], [math.sin(angle_tcp), math.cos(angle_tcp), 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    R_des = R_z @ R_y @ R_x
    p_x, p_y, p_z = p
    r_xy = math.sqrt(p_x ** 2 + p_y ** 2)
    if r_xy < 1e-06:
        q1 = 0.0
    else:
        theta = math.atan2(p_y, p_x)
        ratio = y_const / r_xy
        ratio = max(-1.0, min(1.0, ratio))
        a_angle = math.asin(ratio)
        q1_cand1 = theta - a_angle
        q1_cand2 = theta - (math.pi - a_angle)

        def error_q1(q):
            return abs(-p_x * math.sin(q) + p_y * math.cos(q) - y_const)
        q1 = q1_cand1 if error_q1(q1_cand1) <= error_q1(q1_cand2) else q1_cand2
    cos_q1 = math.cos(q1)
    sin_q1 = math.sin(q1)
    Rz_neg_q1 = np.array([[cos_q1, sin_q1, 0], [-sin_q1, cos_q1, 0], [0, 0, 1]])
    M = Rz_neg_q1 @ R_des
    phi = math.atan2(M[0, 2], M[2, 2])
    p_vec = np.array(p)
    p_bar = Rz_neg_q1 @ p_vec
    P_x = p_bar[0] - L_tcp * math.sin(phi)
    P_z = p_bar[2] - L_tcp * math.cos(phi)
    r2 = math.sqrt(P_x ** 2 + P_z ** 2)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_candA = math.acos(cos_q3)
    q3_candB = -q3_candA

    def planar_solution(q3_val):
        q2_val = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q4_val = phi - (q2_val + q3_val)
        calc_P_x = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val)
        calc_P_z = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val)
        err = math.sqrt((calc_P_x - P_x) ** 2 + (calc_P_z - P_z) ** 2)
        return (q2_val, q4_val, err)
    q2_A, q4_A, err_A = planar_solution(q3_candA)
    q2_B, q4_B, err_B = planar_solution(q3_candB)
    if err_A <= err_B:
        q2, q3, q4 = (q2_A, q3_candA, q4_A)
    else:
        q2, q3, q4 = (q2_B, q3_candB, q4_B)
    q5_prelim = math.atan2(M[1, 0], M[1, 1])
    phi_arm = q2 + q3 + q4
    Rz_q1 = np.array([[math.cos(q1), -math.sin(q1), 0], [math.sin(q1), math.cos(q1), 0], [0, 0, 1]])
    Ry_phi = np.array([[math.cos(phi_arm), 0, math.sin(phi_arm)], [0, 1, 0], [-math.sin(phi_arm), 0, math.cos(phi_arm)]])
    R_arm = Rz_q1 @ Ry_phi
    A = R_arm.T @ R_des @ R_tcp_offset.T
    q6 = math.atan2(-A[2, 0], A[2, 2])
    q5 = math.atan2(A[1, 0], A[0, 0])
    return (q1, q2, q3, q4, q5, q6)