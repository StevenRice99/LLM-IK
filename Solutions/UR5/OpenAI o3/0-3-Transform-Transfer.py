import numpy as np

def _rotx(a: float) -> np.ndarray:
    ca, sa = (np.cos(a), np.sin(a))
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

def _roty(a: float) -> np.ndarray:
    ca, sa = (np.cos(a), np.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

def _rotz(a: float) -> np.ndarray:
    ca, sa = (np.cos(a), np.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return _rotx(roll) @ _roty(pitch) @ _rotz(yaw)

def _matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """
    Inverse of _rpy_to_matrix  (xyz‑fixed axis convention).
    Keeps the three returned angles in the interval  (‑π , π].
    """
    sp = R[0, 2]
    cp = np.sqrt(max(0.0, 1.0 - sp * sp))
    if cp > 1e-08:
        roll = np.arctan2(-R[1, 2], R[2, 2])
        pitch = np.arcsin(sp)
        yaw = np.arctan2(-R[0, 1], R[0, 0])
    else:
        roll = np.arctan2(R[2, 1], R[1, 1])
        pitch = np.pi / 2 * np.sign(sp)
        yaw = 0.0
    return (_wrap(roll), _wrap(pitch), _wrap(yaw))

def _wrap(a: float) -> float:
    """wrap angle to (‑π , π]"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def _fk(q1: float, q2: float, q3: float, q4: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns TCP position (3×1) and orientation (3×3) for the given
    joint values.
    """
    d1 = np.array([0.0, 0.13585, 0.0])
    d2 = np.array([0.0, -0.1197, 0.425])
    d3 = np.array([0.0, 0.0, 0.39225])
    d4 = np.array([0.0, 0.093, 0.0])
    R01 = _rotz(q1)
    p01 = R01 @ d1
    R12 = _roty(q2)
    R02 = R01 @ R12
    p02 = p01 + R02 @ d2
    R23 = _roty(q3)
    R03 = R02 @ R23
    p03 = p02 + R03 @ d3
    R34 = _roty(q4)
    R04 = R03 @ R34
    p04 = p03 + R04 @ d4
    return (p04, R04)

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed‑form inverse kinematics (position + orientation) for the
    4‑DOF manipulator described in the task.

    Parameters
    ----------
    p : (x, y, z) – desired TCP position  [m]
    r : (roll, pitch, yaw) – desired orientation, xyz‑fixed‑axis RPY  [rad]

    Returns
    -------
    (q1, q2, q3, q4)  —\xa0joint values in radians, each wrapped to (‑π , π]
    """
    a = 0.425
    b = 0.39225
    d = 0.10915
    x, y, z = p
    R_d = _rpy_to_matrix(*r)
    c3 = (x * x + y * y + z * z - (a * a + b * b + d * d)) / (2.0 * a * b)
    c3 = np.clip(c3, -1.0, 1.0)
    q3_candidates = (np.arccos(c3), -np.arccos(c3))
    S_mag = np.sqrt(max(x * x + y * y - d * d, 0.0))
    candidates: list[tuple[float, float, float, float]] = []
    for S_sign in (+1, -1):
        S = S_sign * S_mag
        phi = np.arctan2(d, S) if S_mag > 1e-12 else np.pi / 2
        base_ang = np.arctan2(y, x)
        q1 = base_ang - phi
        for q3 in q3_candidates:
            A = a + b * np.cos(q3)
            B = b * np.sin(q3)
            num = S * A - z * B
            den = S * B + z * A
            q2 = np.arctan2(num, den)
            theta = np.arctan2(-R_d[2, 0], R_d[2, 2])
            q4 = theta - q2 - q3
            candidates.append((_wrap(q1), _wrap(q2), _wrap(q3), _wrap(q4)))
    best_err = np.inf
    best_q = candidates[0]
    r_des = np.asarray(r)
    for q1, q2, q3, q4 in candidates:
        p_c, R_c = _fk(q1, q2, q3, q4)
        e_pos = np.linalg.norm(p_c - np.asarray(p))
        r_c = np.asarray(_matrix_to_rpy(R_c))
        e_ori = np.linalg.norm(r_c - r_des)
        err = e_pos + 5.0 * e_ori
        if err < best_err:
            best_err = err
            best_q = (q1, q2, q3, q4)
    return best_q