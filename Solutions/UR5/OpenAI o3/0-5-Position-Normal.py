import numpy as np

def _fk_position(joints: np.ndarray) -> np.ndarray:
    """
    Computes the TCP position for a given 6‑tuple of joint angles.
    The implementation follows the order and dimensions given in the
    DETAILS table of the task description.
    """
    t1, t2, t3, t4, t5, t6 = joints

    def Rz(q):
        c, s = (np.cos(q), np.sin(q))
        return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def Ry(q):
        c, s = (np.cos(q), np.sin(q))
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])

    def T(v):
        out = np.eye(4)
        out[:3, 3] = v
        return out
    T0 = np.eye(4)
    T0 = T0 @ Rz(t1)
    T0 = T0 @ T([0, 0.13585, 0]) @ Ry(t2)
    T0 = T0 @ T([0, -0.1197, 0.425]) @ Ry(t3)
    T0 = T0 @ T([0, 0, 0.39225]) @ Ry(t4)
    T0 = T0 @ T([0, 0.093, 0]) @ Rz(t5)
    T0 = T0 @ T([0, 0, 0.09465]) @ Ry(t6)
    T0 = T0 @ T([0, 0.0823, 0])
    return T0[:3, 3]

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Returns one set of joint angles that places the tool‑centre‑point at
    the requested position p = (x, y, z).  Orientation is not constrained.
    A damped‑least‑squares iterative solver is used; because the problem
    is well conditioned in the robot’s normal workspace only a handful
    of iterations are required.
    """
    target = np.asarray(p, dtype=float)
    q = np.zeros(6)
    eps_pos = 1e-06
    eps_jac = 1e-06
    damping = 0.0001
    max_iter = 100
    for _ in range(max_iter):
        pc = _fk_position(q)
        err = target - pc
        if np.linalg.norm(err) < eps_pos:
            break
        J = np.zeros((3, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += eps_jac
            J[:, i] = (_fk_position(dq) - pc) / eps_jac
        JT = J.T
        JJT = J @ JT
        delta = JT @ np.linalg.solve(JJT + damping * np.eye(3), err)
        q += delta
    q = (q + np.pi) % (2.0 * np.pi) - np.pi
    return tuple((float(v) for v in q))