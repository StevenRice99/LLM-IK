import math
import numpy as np

def _Rx(q):
    c, s = (math.cos(q), math.sin(q))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def _Ry(q):
    c, s = (math.cos(q), math.sin(q))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

def _Rz(q):
    c, s = (math.cos(q), math.sin(q))
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def _T(v):
    out = np.eye(4)
    out[:3, 3] = v
    return out

def _rpy_to_R(rpy):
    """URDF X‑Y‑Z (roll‑pitch‑yaw) to 3×3 rotation"""
    roll, pitch, yaw = rpy
    return (_Rz(yaw) @ _Ry(pitch) @ _Rx(roll))[:3, :3]

def _log_so3(R):
    """
    rotation matrix -> axis–angle vector (length = angle, direction = axis)
    numerically stable for all angles except exactly π (degenerate for robots)
    """
    tr = np.trace(R)
    cos_theta = max(min((tr - 1.0) * 0.5, 1.0), -1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-06:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    coef = theta / (2.0 * math.sin(theta))
    return coef * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def _fk(q):
    """
    complete forward kinematics for position & orientation
    returns (p, R)   p: 3‑vector,   R: 3×3 rotation
    """
    t1, t2, t3, t4, t5, t6 = q
    T = np.eye(4) @ _Rz(t1) @ _T([0, 0.13585, 0]) @ _Ry(t2) @ _T([0, -0.1197, 0.425]) @ _Ry(t3) @ _T([0, 0, 0.39225]) @ _Ry(t4) @ _T([0, 0.093, 0]) @ _Rz(t5) @ _T([0, 0, 0.09465]) @ _Ry(t6) @ _T([0, 0.0823, 0])
    return (T[:3, 3].copy(), T[:3, :3].copy())

def _one_shot_ik(p_des, R_des, q0, pos_tol=1e-06, ori_tol=1e-06, h=1e-05, damping=0.001, it_max=200, ori_weight=0.4):
    """
    Levenberg–Marquardt iteration starting from q0.
    Returns (q, err_norm, success_flag)
    """
    q = q0.copy()
    for _ in range(it_max):
        p_cur, R_cur = _fk(q)
        e_p = p_des - p_cur
        e_ori = _log_so3(R_des @ R_cur.T)
        if np.linalg.norm(e_p) < pos_tol and np.linalg.norm(e_ori) < ori_tol:
            return (q, 0.0, True)
        e = np.hstack((e_p, ori_weight * e_ori))
        J = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += h
            p_dq, R_dq = _fk(dq)
            J[:3, i] = (p_dq - p_cur) / h
            delta_R = _log_so3(R_des @ R_dq.T) - e_ori
            J[3:, i] = ori_weight * delta_R / h
        JT = J.T
        H = J @ JT + damping * np.eye(6)
        dq = JT @ np.linalg.solve(H, e)
        q += dq
    err = np.linalg.norm(e_p) + np.linalg.norm(e_ori)
    return (q, err, False)

def inverse_kinematics(p, r):
    """
    Robust numerical 6‑DOF inverse kinematics.
    Multiple starting guesses are tried; the best convergent solution is
    returned.  Position and orientation are both satisfied to the solver’s
    internal tolerances.
    """
    p_des = np.asarray(p, float)
    R_des = _rpy_to_R(r)
    seeds = [np.zeros(6)]
    corner = [0.0, math.pi]
    for a in corner:
        for b in corner:
            seeds.append(np.array([a, b, 0, 0, 0, 0]))
            seeds.append(np.array([0, 0, a, b, 0, 0]))
    rng = np.random.default_rng(seed=11259375)
    for _ in range(12):
        seeds.append(rng.uniform(-math.pi, math.pi, 6))
    best_q = None
    best_err = float('inf')
    for q0 in seeds:
        q, err, success = _one_shot_ik(p_des, R_des, q0)
        if success:
            best_q = q
            break
        if err < best_err:
            best_err = err
            best_q = q
    q_out = (best_q + math.pi) % (2.0 * math.pi) - math.pi
    return tuple((float(v) for v in q_out))