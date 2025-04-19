import math
import numpy as np

def _wrap(a: float) -> float:
    """wrap angle into (‑π , π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _rpy_to_matrix(r: tuple[float, float, float]) -> np.ndarray:
    """URDF convention:  R = Rz · Ry · Rx."""
    rx, ry, rz = r
    cx, sx = (math.cos(rx), math.sin(rx))
    cy, sy = (math.cos(ry), math.sin(ry))
    cz, sz = (math.cos(rz), math.sin(rz))
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx

def _matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of _rpy_to_matrix (matches ROS / SciPy Z‑Y‑X)."""
    if abs(R[2, 0]) < 1.0 - 1e-12:
        pitch = math.asin(-R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.pi / 2 * (-1 if R[2, 0] > 0 else 1)
        yaw = 0.0
        roll = math.atan2(-R[0, 1], -R[0, 2]) if R[2, 0] > 0 else math.atan2(R[0, 1], R[0, 2])
    return (_wrap(roll), _wrap(pitch), _wrap(yaw))

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed‑form IK for the 4‑DOF manipulator described in the task.

    Parameters
    ----------
    p : (x, y, z) – desired TCP position   [m]
    r : (rx, ry, rz) – desired orientation as roll‑pitch‑yaw [rad]

    Returns
    -------
    θ1, θ2, θ3, θ4   (all wrapped to (‑π , π])
    """
    A = 0.425
    B = 0.39225
    L_Y = 0.10915
    px, py, pz = p
    rx_t, ry_t, rz_t = r
    R_target = _rpy_to_matrix(r)
    r_xy = math.hypot(px, py)
    if r_xy < 1e-12:
        raise ValueError('Target is on the base Z axis; θ1 undefined.')
    psi = math.atan2(py, px)
    alpha = math.asin(max(-1.0, min(1.0, L_Y / r_xy)))
    theta1_options = [_wrap(psi - alpha), _wrap(psi + alpha - math.pi)]
    psi_options = [_wrap(ry_t), _wrap(math.pi - ry_t)]
    best_err = float('inf')
    best_sol = (0.0, 0.0, 0.0, 0.0)
    for theta1 in theta1_options:
        c1, s1 = (math.cos(theta1), math.sin(theta1))
        Xp = c1 * px + s1 * py
        Zp = pz
        d_sq = Xp * Xp + Zp * Zp
        cos_t3 = (d_sq - A * A - B * B) / (2.0 * A * B)
        cos_t3 = max(min(cos_t3, 1.0), -1.0)
        for theta3 in (math.acos(cos_t3), -math.acos(cos_t3)):
            k1 = A + B * math.cos(theta3)
            k2 = B * math.sin(theta3)
            theta2 = _wrap(math.atan2(Xp, Zp) - math.atan2(k2, k1))
            for psi_des in psi_options:
                theta4 = _wrap(psi_des - theta2 - theta3)
                R_cand = _rpy_to_matrix((0.0, psi_des, theta1))
                roll_c, pitch_c, yaw_c = _matrix_to_rpy(R_cand)
                err = abs(_wrap(roll_c - rx_t)) + abs(_wrap(pitch_c - ry_t)) + abs(_wrap(yaw_c - rz_t))
                X_check = A * math.sin(theta2) + B * math.sin(theta2 + theta3)
                Z_check = A * math.cos(theta2) + B * math.cos(theta2 + theta3)
                pos_err = math.hypot(X_check - Xp, Z_check - Zp)
                err += 0.001 * pos_err
                if err < best_err:
                    best_err = err
                    best_sol = (_wrap(theta1), _wrap(theta2), _wrap(theta3), _wrap(theta4))
    return best_sol