import math
import numpy as np

def Rx(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

def Ry(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

def Rz(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

def _solve_first_five(p: np.ndarray, R5: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Analytic solution for joints q1 … q5 exactly as derived in the EXISTING section,
    but expressed directly with a rotation matrix (R5) instead of roll‑pitch‑yaw.
    """
    L1 = 0.425
    L2 = 0.39225
    L_tcp = 0.09465
    y_const = 0.13585 - 0.1197 + 0.093
    p_x, p_y, p_z = (float(p[0]), float(p[1]), float(p[2]))
    r_xy = math.hypot(p_x, p_y)
    theta = math.atan2(p_y, p_x)
    ratio = y_const / r_xy
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_cand = (theta - a, theta - (math.pi - a))

    def _error_q1(q1_val: float) -> float:
        c1, s1 = (math.cos(q1_val), math.sin(q1_val))
        Rz_m = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        M = Rz_m @ R5
        return abs(M[1, 2])
    q1 = min(q1_cand, key=_error_q1)
    c1, s1 = (math.cos(q1), math.sin(q1))
    Rz_m = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    M = Rz_m @ R5
    phi = math.atan2(M[0, 2], M[2, 2])
    q5 = math.atan2(M[1, 0], M[1, 1])
    p_bar = Rz_m @ np.array([p_x, p_y, p_z])
    P_x = p_bar[0] - L_tcp * math.sin(phi)
    P_z = p_bar[2] - L_tcp * math.cos(phi)
    r2 = math.hypot(P_x, P_z)
    cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_cand = (math.acos(cos_q3), -math.acos(cos_q3))

    def _planar_branch(q3_val: float):
        q2_val = math.atan2(P_x, P_z) - math.atan2(L2 * math.sin(q3_val), L1 + L2 * math.cos(q3_val))
        q4_val = phi - (q2_val + q3_val)
        calc_x = L1 * math.sin(q2_val) + L2 * math.sin(q2_val + q3_val) + L_tcp * math.sin(phi)
        calc_z = L1 * math.cos(q2_val) + L2 * math.cos(q2_val + q3_val) + L_tcp * math.cos(phi)
        err = math.hypot(calc_x - p_bar[0], calc_z - p_bar[2])
        return (q2_val, q4_val, err)
    solA = _planar_branch(q3_cand[0])
    solB = _planar_branch(q3_cand[1])
    if solA[2] <= solB[2]:
        q3, q2, q4 = (q3_cand[0], solA[0], solA[1])
    else:
        q3, q2, q4 = (q3_cand[1], solB[0], solB[1])
    return (q1, q2, q3, q4, q5)

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‑form analytical inverse kinematics for the 6‑DOF arm described
    in the DETAILS section.

    Parameters
    ----------
    p : (x, y, z)
        Desired TCP position in metres, expressed in the base frame.
    r : (roll, pitch, yaw)
        Desired TCP orientation in radians (URDF convention: Rz(yaw)·Ry(pitch)·Rx(roll)).

    Returns
    -------
    (q1 … q6)  : joint angles in radians.
    """
    d6 = 0.09465
    tcp_dy = 0.0823
    roll, pitch, yaw = r
    R_des = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    q6 = 0.0
    for _ in range(3):
        R5_target = R_des @ Rz(-math.pi / 2) @ Ry(-q6)
        vec_56 = np.array([0.0, 0.0, d6])
        vec_6tcp = Ry(q6) @ np.array([0.0, tcp_dy, 0.0])
        p6_target = np.array(p) - R5_target @ (vec_56 + vec_6tcp)
        q1, q2, q3, q4, q5 = _solve_first_five(p6_target, R5_target)
        phi = q2 + q3 + q4
        R5_current = Rz(q1) @ Ry(phi) @ Rz(q5)
        R_rel = R5_current.T @ (R_des @ Rz(-math.pi / 2))
        q6_new = math.atan2(R_rel[0, 2], R_rel[2, 2])
        if abs(q6_new - q6) < 1e-09:
            q6 = q6_new
            break
        q6 = q6_new
    return (q1, q2, q3, q4, q5, q6)