import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‑form inverse kinematics for the complete 6‑DOF manipulator.

    Parameters
    ----------
    p : (x, y, z)
        Desired TCP position in the base frame (metres).
    r : (roll, pitch, yaw)
        Desired TCP orientation given as intrinsic XYZ (URDF convention) Euler
        angles, in radians.

    Returns
    -------
    q1 … q6 : float
        A consistent joint configuration (radians) that realises (p, r).
    """
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    TCP_Y = 0.0823
    y_const = 0.13585 - 0.1197 + 0.093

    def Rx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def Ry(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_des = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    p_vec = np.array(p, dtype=float)
    p6_vec = p_vec - R_des @ np.array([TCP_Y, 0.0, 0.0])
    p6_x, p6_y, p6_z = p6_vec
    r_xy = math.hypot(p6_x, p6_y)
    ratio = max(-1.0, min(1.0, y_const / max(1e-09, r_xy)))
    theta = math.atan2(p6_y, p6_x)
    a = math.asin(ratio)
    q1_candidates = (theta - a, theta - (math.pi - a))
    best_err = float('inf')
    best_sol = None
    for q1 in q1_candidates:
        cq1, sq1 = (math.cos(q1), math.sin(q1))
        Rz_m_q1 = np.array([[cq1, sq1, 0], [-sq1, cq1, 0], [0, 0, 1]])
        R_tmp = Rz_m_q1 @ R_des @ Rz(-math.pi / 2)
        s5 = math.hypot(R_tmp[1, 0], R_tmp[1, 2])
        c5 = R_tmp[1, 1]
        q5 = math.atan2(R_tmp[1, 0], c5)
        if abs(s5) < 1e-08:
            q5 = 0.0
            q6 = 0.0
            phi = math.atan2(R_tmp[0, 2], R_tmp[2, 2])
        else:
            q6 = math.atan2(R_tmp[1, 2], R_tmp[1, 0])
            phi = math.atan2(R_tmp[2, 1], -R_tmp[0, 1])
        p_bar = Rz_m_q1 @ p6_vec
        pb_x, pb_y, pb_z = p_bar
        Px = pb_x - L_tcp * math.sin(phi)
        Pz = pb_z - L_tcp * math.cos(phi)
        r2 = math.hypot(Px, Pz)
        cos_q3 = (r2 * r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        for q3 in (math.acos(cos_q3), -math.acos(cos_q3)):
            s3 = math.sin(q3)
            c3 = math.cos(q3)
            q2 = math.atan2(Px, Pz) - math.atan2(L2 * s3, L1 + L2 * c3)
            q4 = phi - (q2 + q3)
            calc_x = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) + L_tcp * math.sin(phi)
            calc_z = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) + L_tcp * math.cos(phi)
            err = math.hypot(calc_x - pb_x, calc_z - pb_z)
            if err < best_err:
                best_err = err
                best_sol = (q1, q2, q3, q4, q5, q6)
    if best_sol is None:
        raise RuntimeError('IK solver failed to converge.')
    return best_sol