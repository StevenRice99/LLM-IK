import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Analytic 5‑DOF IK with full 2×2 branch enumeration + in‑code forward kinematics
    to pick the unique solution that actually reaches (p,r).
    """
    px, py, pz = p
    roll, pitch, yaw = r
    L1, L2 = (0.425, 0.39225)
    d4, d5 = (0.09465, 0.0823)

    def rotx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def roty(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rotz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def roty4(a):
        H = np.eye(4)
        ca, sa = (np.cos(a), np.sin(a))
        H[0, 0], H[0, 2] = (ca, sa)
        H[2, 0], H[2, 2] = (-sa, ca)
        return H

    def rotz4(a):
        H = np.eye(4)
        ca, sa = (np.cos(a), np.sin(a))
        H[0, 0], H[0, 1] = (ca, -sa)
        H[1, 0], H[1, 1] = (sa, ca)
        return H

    def trans(v):
        T = np.eye(4)
        T[:3, 3] = v
        return T
    R_des = rotz(yaw) @ roty(pitch) @ rotx(roll)
    R0_5 = R_des @ rotz(-0.5 * np.pi)
    M = R0_5
    phi = np.arctan2(M[2, 1], -M[0, 1])
    c4 = np.clip(M[1, 1], -1.0, 1.0)
    theta = np.arccos(c4)
    psi = np.arctan2(M[1, 2], M[1, 0])
    branches = [(phi, theta, psi), (phi + np.pi, -theta, psi + np.pi)]
    pd = np.array([px, py, pz])
    p0_5 = pd - R0_5 @ np.array([0.0, d5, 0.0])
    best_sol = None
    best_err = 1000000000.0
    for S, q4, q5 in branches:
        R0_4 = roty(S) @ rotz(q4)
        p0_4 = p0_5 - R0_4 @ np.array([0.0, 0.0, d4])
        x4, z4 = (p0_4[0], p0_4[2])
        D = (x4 * x4 + z4 * z4 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        if abs(D) > 1.0:
            continue
        for sign in (+1.0, -1.0):
            q2 = sign * np.arccos(np.clip(D, -1.0, 1.0))
            phi2 = np.arctan2(x4, z4)
            psi2 = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
            q1 = phi2 - psi2
            q3 = S - q1 - q2
            T = np.eye(4)
            T = T @ roty4(q1) @ trans([0, -0.1197, 0.425])
            T = T @ roty4(q2) @ trans([0, 0.0, 0.39225])
            T = T @ roty4(q3) @ trans([0, 0.093, 0.0])
            T = T @ rotz4(q4) @ trans([0, 0.0, 0.09465])
            T = T @ roty4(q5) @ trans([0, 0.0823, 0.0]) @ rotz4(0.5 * np.pi)
            p_pred = T[:3, 3]
            R_pred = T[:3, :3]
            err_p = np.linalg.norm(p_pred - pd)
            R_err = R_pred.T @ R_des
            ang = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
            err = err_p + ang * 0.1
            if err < best_err:
                best_err = err
                best_sol = (q1, q2, q3, q4, q5)
    if best_sol is None:
        raise ValueError('IK: no valid solution found')

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    return tuple((wrap(x) for x in best_sol))