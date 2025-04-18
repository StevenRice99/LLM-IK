import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Analytic IK for the 5‑DOF serial arm (position only).
    Returns an “elbow‑down” solution [q1..q5] in radians.
    """
    x, y, z = p
    d1 = 0.13585
    d2 = -0.1197
    d3 = 0.39225
    d4 = 0.093
    d5 = 0.09465
    L1 = 0.425
    L2 = d3 + d5
    d_total = d1 + d2 + d4
    r = np.hypot(x, y)
    φ = np.arctan2(-x, y)
    arg = np.clip(d_total / r, -1.0, 1.0)
    θ = np.arccos(arg)
    q1 = φ - θ
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    c1 = np.cos(q1)
    s1 = np.sin(q1)
    x1 = c1 * x + s1 * y
    z1 = z
    D = (x1 * x1 + z1 * z1 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)
    q3 = -np.arccos(D)
    A = L1 + L2 * np.cos(q3)
    B = L2 * np.sin(q3)
    q2 = np.arctan2(x1 * A - z1 * B, z1 * A + x1 * B)

    def Rz(th):
        return np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])

    def Ry(th):
        return np.array([[np.cos(th), 0.0, np.sin(th)], [0.0, 1.0, 0.0], [-np.sin(th), 0.0, np.cos(th)]])
    R1 = Rz(q1)
    R2 = R1 @ Ry(q2)
    R3 = R2 @ Ry(q3)
    p1 = R1 @ np.array([0.0, d1, 0.0])
    p2 = R2 @ np.array([0.0, d2, L1])
    p3 = R3 @ np.array([0.0, 0.0, d3])
    origin3 = p1 + p2 + p3
    v3 = np.array([x, y, z]) - origin3
    pl = R3.T @ v3
    q4 = np.arctan2(pl[0], pl[2])
    q5 = 0.0

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    return (wrap(q1), wrap(q2), wrap(q3), wrap(q4), wrap(q5))