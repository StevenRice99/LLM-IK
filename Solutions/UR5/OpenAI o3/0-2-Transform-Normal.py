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

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Closed‑form inverse kinematics for the 3‑DOF serial arm described in
    the DETAILS section (axes Z‑Y‑Y).

    Parameters
    ----------
    p : (x, y, z)
        Desired TCP position expressed in the base frame [m].
    r : (roll, pitch, yaw)
        Desired TCP orientation given as intrinsic X‑Y‑Z Euler angles
        (roll–pitch–yaw) [rad].

    Returns
    -------
    (θ1, θ2, θ3) :
        Joint values for Revolute‑1, ‑2 and ‑3 [rad].
    """
    d1 = np.array([0.0, 0.13585, 0.0])
    d2 = np.array([0.0, -0.1197, 0.425])
    d3 = np.array([0.0, 0.0, 0.39225])
    a = d3[2]
    b = d2[2]
    const_y = d2[1]
    roll, pitch, yaw = r
    R_d = _rotz(yaw) @ _roty(pitch) @ _rotx(roll)
    theta23_des = np.arctan2(-R_d[2, 0], R_d[2, 2])
    theta1 = np.arctan2(-R_d[0, 1], R_d[1, 1])
    p = np.asarray(p, dtype=float)
    p1 = _rotz(-theta1) @ p - d1
    px, py, pz = p1
    r2 = px * px + pz * pz
    r = np.sqrt(r2)
    cos_t3 = (r2 - (a * a + b * b)) / (2.0 * a * b)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    theta3_candidates = np.array([np.arccos(cos_t3), -np.arccos(cos_t3)])

    def _pos_from_2(th2: float, th3: float) -> np.ndarray:
        """
        Fast evaluation of the vector joint‑2 → TCP in frame {1}.
        """
        c2, s2 = (np.cos(th2), np.sin(th2))
        c3, s3 = (np.cos(th3), np.sin(th3))
        k1 = b + a * c3
        k2 = a * s3
        vec_x = c2 * k2 + s2 * k1
        vec_z = -s2 * k2 + c2 * k1
        return np.array([vec_x, const_y, vec_z])
    best_err = np.inf
    best_sol = (0.0, 0.0, 0.0)
    alpha = np.arctan2(px, pz)
    for th3 in theta3_candidates:
        s3 = np.sin(th3)
        c3 = np.cos(th3)
        gamma = np.arctan2(a * s3, b + a * c3)
        th2 = alpha - gamma
        pred = _pos_from_2(th2, th3)
        pos_err = np.linalg.norm(pred - p1)
        ori_err = abs((th2 + th3 - theta23_des + np.pi) % (2 * np.pi) - np.pi)
        total_err = pos_err + ori_err
        if total_err < best_err:
            best_err = total_err
            best_sol = (th2, th3)
    theta2, theta3 = best_sol
    wrap = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi
    theta1 = wrap(theta1)
    theta2 = wrap(theta2)
    theta3 = wrap(theta3)
    return (float(theta1), float(theta2), float(theta3))