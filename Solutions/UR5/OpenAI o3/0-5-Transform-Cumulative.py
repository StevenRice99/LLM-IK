import math
import numpy as np

def _rot_x(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

def _rot_y(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

def _rot_z(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Closed‑form analytic inverse kinematics for the 6‑DOF manipulator
    described in the prompt.

    Parameters
    ----------
    p : (x, y, z)
        Desired position of the TCP expressed in the base frame [m].
    r : (roll, pitch, yaw)
        Desired orientation of the TCP given as fixed‑axis intrinsic
        xyz–RPY angles [rad].

    Returns
    -------
    (q1 … q6) : Six joint angles in radians that realise the requested pose.
    """
    L1 = 0.425
    L2 = 0.39225
    L6 = 0.09465
    Y12 = 0.13585
    Y23 = -0.1197
    Y45 = 0.093
    Ytcp = 0.0823
    SIGMA = math.pi / 2.0
    Y_CONST = Y12 + Y23 + Y45
    L_TCP = L6
    roll, pitch, yaw = r
    R_des = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    p_x, p_y, p_z = p
    theta = math.atan2(p_y, p_x)
    r_xy = math.hypot(p_x, p_y)
    ratio = Y_CONST / max(r_xy, 1e-09)
    ratio = max(-1.0, min(1.0, ratio))
    a = math.asin(ratio)
    q1_cand = (theta - a, theta - (math.pi - a))

    def _branch(q1_val: float) -> tuple[float, float, float, float, float, float, float]:
        c1, s1 = (math.cos(q1_val), math.sin(q1_val))
        Rz_neg_q1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
        M = Rz_neg_q1 @ R_des @ _rot_z(-SIGMA)
        s5 = math.hypot(M[1, 0], M[1, 2])
        q5 = math.atan2(s5, M[1, 1])
        if M[1, 1] < 0:
            q5 = math.copysign(math.pi - q5, M[1, 0])
        phi = math.atan2(M[2, 1], -M[0, 1])
        q6 = math.atan2(M[1, 2], M[1, 0])
        R01 = _rot_z(q1_val)
        R04 = R01 @ _rot_y(phi)
        Rz_q5 = _rot_z(q5)
        Ry_q6 = _rot_y(q6)
        offset_4 = np.array([0, Y45, 0]) + Rz_q5 @ (np.array([0, 0, L6]) + Ry_q6 @ np.array([0, Ytcp, 0]))
        p_vec = np.array([p_x, p_y, p_z])
        p4 = p_vec - R04 @ offset_4
        p4_bar = Rz_neg_q1 @ p4
        px_bar, py_bar, pz_bar = p4_bar
        L_eff_x = px_bar
        L_eff_z = pz_bar
        r2 = math.hypot(L_eff_x, L_eff_z)
        cos_q3 = (r2 ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3a = math.acos(cos_q3)
        q3b = -q3a
        best = None
        for q3 in (q3a, q3b):
            k1 = L1 + L2 * math.cos(q3)
            k2 = L2 * math.sin(q3)
            q2 = math.atan2(L_eff_x, L_eff_z) - math.atan2(k2, k1)
            q4 = phi - q2 - q3
            x_chk = L1 * math.sin(q2) + L2 * math.sin(q2 + q3)
            z_chk = L1 * math.cos(q2) + L2 * math.cos(q2 + q3)
            err = math.hypot(x_chk - L_eff_x, z_chk - L_eff_z)
            if best is None or err < best[-1]:
                best = (q2, q3, q4, err)
        q2, q3, q4, err_pos = best
        return (q2, q3, q4, q5, q6, phi, err_pos)
    sol_A = _branch(q1_cand[0])
    sol_B = _branch(q1_cand[1])
    if sol_A[-1] <= sol_B[-1]:
        q1 = q1_cand[0]
        q2, q3, q4, q5, q6, _phi, _ = sol_A
    else:
        q1 = q1_cand[1]
        q2, q3, q4, q5, q6, _phi, _ = sol_B
    return (q1, q2, q3, q4, q5, q6)