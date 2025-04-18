import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Analytical inverse kinematics for the 6‑DOF UR robot described.
    Joint axes: Z, Y, Y, Y, Z, Y.
    Link‑origin offsets (in order) given in the URDF:
      d2_y = 0.13585
      d3_y = -0.1197, d3_z = 0.425
      d4_z = 0.39225
      d5_y = 0.093
      d6_z = 0.09465
      tcp_y = 0.0823, tcp_orientation_offset about Z = +pi/2
    """
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    psi = 1.570796325
    px, py, pz = p
    roll, pitch, yaw = r
    r_xy = math.hypot(px, py)
    theta = math.atan2(py, px)
    ratio = y_const / r_xy
    ratio = max(-1.0, min(1.0, ratio))
    alpha = math.asin(ratio)
    q1_cand = [theta - alpha, theta - (math.pi - alpha)]

    def RotZ(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def RotY(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def RotX(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    R_des = RotZ(yaw) @ RotY(pitch) @ RotX(roll)
    best = None
    for q1 in q1_cand:
        c1, s1 = (math.cos(q1), math.sin(q1))
        Rz_neg1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        M = Rz_neg1 @ R_des
        err = abs(M[1, 2])
        if best is None or err < best[0]:
            best = (err, q1, M)
    _, q1, M = best
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = math.atan2(M[1, 0], M[1, 1])
    c1, s1 = (math.cos(q1), math.sin(q1))
    Rz_neg1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    p_bar = Rz_neg1 @ np.array([px, py, pz])
    Px = p_bar[0] - L_tcp * math.sin(phi)
    Pz = p_bar[2] - L_tcp * math.cos(phi)
    r2 = math.hypot(Px, Pz)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_candidates = [math.acos(cos_q3), -math.acos(cos_q3)]

    def planar_solution(q3):
        num = L2 * math.sin(q3)
        den = L1 + L2 * math.cos(q3)
        q2 = math.atan2(Px, Pz) - math.atan2(num, den)
        q4 = phi - (q2 + q3)
        x_rec = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) + L_tcp * math.sin(phi)
        z_rec = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) + L_tcp * math.cos(phi)
        err = math.hypot(x_rec - Px - L_tcp * math.sin(phi) + L_tcp * math.sin(phi), z_rec - Pz - L_tcp * math.cos(phi) + L_tcp * math.cos(phi))
        return (q2, q4, err)
    best2 = None
    for q3 in q3_candidates:
        q2, q4, err = planar_solution(q3)
        if best2 is None or err < best2[0]:
            best2 = (err, q2, q3, q4)
    _, q2, q3, q4 = best2
    R01 = RotZ(q1)
    R12 = RotY(q2)
    R23 = RotY(q3)
    R34 = RotY(q4)
    R45 = RotZ(q5)
    R05 = R01 @ R12 @ R23 @ R34 @ R45
    M_rem = R05.T @ R_des @ RotZ(-psi)
    q6 = math.atan2(M_rem[0, 2], M_rem[2, 2])
    return (q1, q2, q3, q4, q5, q6)