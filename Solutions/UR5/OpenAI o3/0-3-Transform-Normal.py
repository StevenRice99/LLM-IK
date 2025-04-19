import math
import numpy as np
A1 = 0.13585
A2 = -0.1197
A4 = 0.093
L1 = 0.425
L2 = 0.39225
CONST_Y = A1 + A2 + A4
LIMIT = 2 * math.pi
TOL = 1e-10

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ Fixed‑axis XYZ (URDF style):  R = Rz(yaw) · Ry(pitch) · Rx(roll) """
    sr, cr = (math.sin(roll), math.cos(roll))
    sp, cp = (math.sin(pitch), math.cos(pitch))
    sy, cy = (math.sin(yaw), math.cos(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

def _matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of  _rpy_to_matrix  (matches tf.transformations.euler_from_matrix)."""
    if abs(R[2, 0]) < 1.0 - 1e-12:
        pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.pi / 2 if R[2, 0] < 0 else -math.pi / 2
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return (roll, pitch, yaw)

def _wrap(a: float) -> float:
    """wrap angle to (‑π, π]"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _angle_err(a: float, b: float) -> float:
    """minimum absolute difference between two angles"""
    d = abs(a - b)
    return d if d <= math.pi else 2.0 * math.pi - d

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Analytical IK for the 4‑DoF manipulator from the task description.

    Parameters
    ----------
    p : (x, y, z) – TCP position (metres) in the base frame
    r : (roll, pitch, yaw) – desired orientation as fixed‑axis RPY (radians)

    Returns
    -------
    (θ₁, θ₂, θ₃, θ₄)  : tuple[float, float, float, float]
        Joint angles that reproduce the requested pose *as closely as the
        kinematic structure allows*.  All values respect the ±2π limits.
    """
    xt, yt, zt = p
    roll_t, pitch_t, yaw_t = r
    R_target = _rpy_to_matrix(roll_t, pitch_t, yaw_t)
    phi_raw = math.atan2(-R_target[2, 0], R_target[2, 2])
    yaw_raw = math.atan2(R_target[1, 0], R_target[0, 0])
    yaw_cands = (yaw_raw, _wrap(yaw_raw + math.pi))
    phi_cands = (phi_raw, _wrap(math.pi - phi_raw))
    best_cost = float('inf')
    best_joints = None
    for yaw_c, phi_c in zip(yaw_cands, phi_cands):
        θ1 = _wrap(yaw_c)
        c1, s1 = (math.cos(θ1), math.sin(θ1))
        x_p = c1 * xt + s1 * yt
        y_p = -s1 * xt + c1 * yt
        z_p = zt
        if abs(y_p - CONST_Y) > 1e-06:
            continue
        D_sq = x_p * x_p + z_p * z_p
        D = math.sqrt(D_sq)
        cos_t3 = (D_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        if abs(cos_t3) > 1.0 + 1e-09:
            continue
        cos_t3 = max(-1.0, min(1.0, cos_t3))
        sin_t3_base = math.sqrt(max(0.0, 1.0 - cos_t3 * cos_t3))
        for elbow_sign in (+1.0, -1.0):
            sin_t3 = elbow_sign * sin_t3_base
            θ3 = math.atan2(sin_t3, cos_t3)
            k1 = L1 + L2 * cos_t3
            k2 = L2 * sin_t3
            γ = math.atan2(x_p, z_p)
            δ = math.atan2(k2, k1)
            θ2 = γ - δ
            θ4 = phi_c - θ2 - θ3
            θ2 = _wrap(θ2)
            θ3 = _wrap(θ3)
            θ4 = (θ4 + math.pi) % (2.0 * math.pi) - math.pi
            φ_total = θ2 + θ3 + θ4
            R_fk = _rpy_to_matrix(0.0, 0.0, θ1) @ _rpy_to_matrix(0.0, φ_total, 0.0)
            roll_fk, pitch_fk, yaw_fk = _matrix_to_rpy(R_fk)
            err_roll = _angle_err(roll_fk, roll_t)
            err_pitch = _angle_err(pitch_fk, pitch_t)
            err_yaw = _angle_err(yaw_fk, yaw_t)
            orient_err = err_roll + err_pitch + err_yaw
            wrap_penalty = 0.0
            if roll_t * roll_fk < 0 and abs(abs(roll_t) - math.pi) < 1e-06:
                wrap_penalty = 10.0
            cost = orient_err + wrap_penalty
            if cost + TOL < best_cost:
                best_cost = cost
                best_joints = (θ1, θ2, θ3, θ4)
    if best_joints is None:
        raise RuntimeError('No IK solution found although target declared reachable')
    θ1, θ2, θ3, θ4 = best_joints
    φ_tot = θ2 + θ3 + θ4
    R_fk = _rpy_to_matrix(0.0, 0.0, θ1) @ _rpy_to_matrix(0.0, φ_tot, 0.0)
    roll_fk, *_ = _matrix_to_rpy(R_fk)
    if _angle_err(roll_fk, roll_t) > _angle_err(_wrap(roll_fk + 2.0 * math.pi), roll_t):
        θ4 += 2.0 * math.pi
    θ1, θ2, θ3, θ4 = map(_wrap, (θ1, θ2, θ3, θ4))
    return (θ1, θ2, θ3, θ4)