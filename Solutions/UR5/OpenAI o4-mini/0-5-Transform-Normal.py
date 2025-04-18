def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    import numpy as np
    from math import sin, cos, acos, atan2, sqrt, pi
    px, py, pz = p
    rx, ry, rz = r

    def rot_x(a):
        return np.array([[1, 0, 0], [0, cos(a), -sin(a)], [0, sin(a), cos(a)]])

    def rot_y(a):
        return np.array([[cos(a), 0, sin(a)], [0, 1, 0], [-sin(a), 0, cos(a)]])

    def rot_z(a):
        return np.array([[cos(a), -sin(a), 0], [sin(a), cos(a), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)

    def fk(q):
        q1, q2, q3, q4, q5, q6 = q
        R01 = rot_z(q1)
        p01 = np.zeros(3)
        t12 = np.array([0, 0.13585, 0])
        R12 = rot_y(q2)
        R02 = R01 @ R12
        p02 = p01 + R01 @ t12
        t23 = np.array([0, -0.1197, 0.425])
        R23 = rot_y(q3)
        R03 = R02 @ R23
        p03 = p02 + R02 @ t23
        t34 = np.array([0, 0, 0.39225])
        R34 = rot_y(q4)
        R04 = R03 @ R34
        p04 = p03 + R03 @ t34
        t45 = np.array([0, 0.093, 0])
        R45 = rot_z(q5)
        R05 = R04 @ R45
        p05 = p04 + R04 @ t45
        t56 = np.array([0, 0, 0.09465])
        R56 = rot_y(q6)
        R06 = R05 @ R56
        p06 = p05 + R05 @ t56
        tcp_t = np.array([0, 0.0823, 0])
        tcp_R = rot_z(pi / 2)
        pos = p06 + R06 @ tcp_t
        R = R06 @ tcp_R
        return (pos, R)

    def orientation_error(R_cur):
        R_e = R_cur.T @ R_target
        tr = max(-1.0, min(3.0, np.trace(R_e)))
        theta = acos((tr - 1.0) / 2.0)
        if abs(theta) < 1e-08:
            return np.array([R_e[2, 1] - R_e[1, 2], R_e[0, 2] - R_e[2, 0], R_e[1, 0] - R_e[0, 1]]) * 0.5
        else:
            return theta / (2.0 * sin(theta)) * np.array([R_e[2, 1] - R_e[1, 2], R_e[0, 2] - R_e[2, 0], R_e[1, 0] - R_e[0, 1]])
    q = np.zeros(6)
    tol = 1e-06
    max_iters = 50
    eps = 1e-06
    for _ in range(max_iters):
        pos_cur, R_cur = fk(q)
        e_p = pos_cur - np.array([px, py, pz])
        e_ori = orientation_error(R_cur)
        err = np.hstack((e_p, e_ori))
        if np.linalg.norm(err) < tol:
            break
        J = np.zeros((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = eps
            p2, R2 = fk(q + dq)
            e2_p = p2 - np.array([px, py, pz])
            e2_o = orientation_error(R2)
            J[:, j] = (np.hstack((e2_p, e2_o)) - err) / eps
        Δq, *_ = np.linalg.lstsq(J, -err, rcond=None)
        q += Δq
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5]))