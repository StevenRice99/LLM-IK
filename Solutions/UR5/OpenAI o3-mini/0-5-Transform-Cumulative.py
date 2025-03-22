import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes a closed-form analytical inverse kinematics solution for the 6-DOF manipulator.

    Robot structure (all units in meters and radians):
      • Revolute 1: at [0, 0, 0], rotates about Z.
      • Revolute 2: translate [0, 0.13585, 0] then rotate about Y.
      • Revolute 3: translate [0, -0.1197, 0.425] then rotate about Y.
      • Revolute 4: translate [0, 0, 0.39225] then rotate about Y.
      • Revolute 5: translate [0, 0.093, 0] then rotate about Z.
      • Revolute 6: translate [0, 0, 0.09465] then rotate about Y.
      • TCP: translate [0, 0.0823, 0] with a fixed yaw offset ψ = 1.570796325 (i.e. an extra Rz(ψ)).
    
    Kinematic decoupling:
      1. The base subproblem uses the constant offset
             y_const = 0.13585 - 0.1197 + 0.093 = 0.10915,
         so that in the base XY–plane a rotation q1 is chosen so that when the target is rotated by –q1,
         its Y-value is (approximately) y_const.
      2. In the rotated frame the desired orientation (constructed via R_des = Rz(yaw)*Ry(pitch)*Rx(roll))
         is decoupled:
              M = Rz(–q1)*R_des.
         One then extracts:
              φ = q2+q3+q4 = atan2(M[0,2], M[2,2]),
         and the wrist joint about Z is
              q5 = atan2(M[1,0], M[1,1]).
      3. With the TCP offset L_tcp = 0.09465 along the direction φ (and the extra TCP translation [0,0.0823,0]),
         the effective planar 2R subchain (with link lengths L1 = 0.425 and L2 = 0.39225) gives q2, q3, and q4.
      4. Finally, the remaining wrist rotation q6 is recovered by noticing that the complete rotation
         satisfies:
              R_total = Rz(q1) · [Ry(q2+q3+q4) · Rz(q5)] · [Ry(q6) · Rz(ψ)] = R_des.
         Hence,
              Ry(q6) = (Rz(q1)·Ry(q2+q3+q4)·Rz(q5))^T · R_des · Rz(–ψ).

    Parameters:
      p: The target TCP position as (x, y, z).
      r: The target TCP orientation in roll, pitch, yaw (URDF convention: Rz(yaw)*Ry(pitch)*Rx(roll)).

    Returns:
      A tuple of joint angles (q1, q2, q3, q4, q5, q6) in radians.
    """

    def norm_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def rot_x(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    def rot_y(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def rot_z(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    psi = 1.570796325
    p_x, p_y, p_z = p
    p_vec = np.array([p_x, p_y, p_z])
    roll, pitch, yaw = r
    R_des = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    r_xy = math.sqrt(p_x ** 2 + p_y ** 2)
    if r_xy < 1e-08:
        theta = 0.0
    else:
        theta = math.atan2(p_y, p_x)
    ratio = y_const / r_xy if r_xy != 0 else 0.0
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_cand1 = norm_angle(theta - a)
    q1_cand2 = norm_angle(theta - (math.pi - a))

    def compute_M(q1_val):
        return rot_z(-q1_val) @ R_des
    M1 = compute_M(q1_cand1)
    M2 = compute_M(q1_cand2)
    err1 = abs(M1[1, 2])
    err2 = abs(M2[1, 2])
    q1 = norm_angle(q1_cand1 if err1 <= err2 else q1_cand2)
    Rz_neg_q1 = rot_z(-q1)
    p_bar = Rz_neg_q1 @ p_vec
    p_bar_x, p_bar_y, p_bar_z = p_bar
    M = Rz_neg_q1 @ R_des
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = norm_angle(math.atan2(M[1, 0], M[1, 1]))
    P_x = p_bar_x - L_tcp * math.sin(phi)
    P_z = p_bar_z - L_tcp * math.cos(phi)
    r2 = math.sqrt(P_x ** 2 + P_z ** 2)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_candA = math.acos(cos_q3)
    q3_candB = -q3_candA

    def solve_planar(q3_val):
        q2_val = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q4_val = norm_angle(phi - (q2_val + q3_val))
        calc_x = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L_tcp * math.sin(phi)
        calc_z = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L_tcp * math.cos(phi)
        err = math.sqrt((calc_x - p_bar_x) ** 2 + (calc_z - p_bar_z) ** 2)
        return (norm_angle(q2_val), norm_angle(q4_val), err)
    q2_A, q4_A, err_A = solve_planar(q3_candA)
    q2_B, q4_B, err_B = solve_planar(q3_candB)
    if err_A <= err_B:
        q3 = norm_angle(q3_candA)
        q2 = q2_A
        q4 = q4_A
    else:
        q3 = norm_angle(q3_candB)
        q2 = q2_B
        q4 = q4_B
    R1 = rot_z(q1)
    R_y_total = rot_y(q2 + q3 + q4)
    Rz_q5 = rot_z(q5)
    R_mid = R1 @ R_y_total @ Rz_q5
    R_rem = R_mid.T @ R_des
    A = R_rem @ rot_z(-psi)
    q6 = norm_angle(math.atan2(A[0, 2], A[0, 0]))
    q1 = norm_angle(q1)
    q2 = norm_angle(q2)
    q3 = norm_angle(q3)
    q4 = norm_angle(q4)
    q5 = norm_angle(q5)
    q6 = norm_angle(q6)
    return (q1, q2, q3, q4, q5, q6)