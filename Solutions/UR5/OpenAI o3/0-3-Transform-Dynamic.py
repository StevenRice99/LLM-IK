import math
import numpy as np
A1 = 0.13585
A2 = -0.1197
B2 = 0.425
B3 = 0.39225
A4 = 0.093
Y_CONST = A1 + A2 + A4

def _rpy_to_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Intrinsic X‑Y‑Z (roll‑pitch‑yaw) → rotation matrix."""
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

def _clip_to_range(a: float, low: float, high: float) -> float:
    return max(low, min(high, a))

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Analytical 4‑DOF IK for the arm described in the task.
    Returns joint values (q1, q2, q3, q4) in radians – each constrained
    only to the legal interval [‑2\xa0π,\xa02\xa0π] but never wrapped to (‑π,\xa0π].
    """
    R_goal = _rpy_to_mat(*r)
    q1 = math.atan2(-R_goal[0, 1], R_goal[1, 1])
    theta_sum = math.atan2(-R_goal[2, 0], R_goal[2, 2])
    c1, s1 = (math.cos(q1), math.sin(q1))
    x_w, y_w, z_w = p
    x1 = c1 * x_w + s1 * y_w
    y1 = -s1 * x_w + c1 * y_w
    z1 = z_w
    y_error = y1 - Y_CONST
    L_sq = x1 * x1 + z1 * z1
    L = math.sqrt(L_sq)
    cos_q3 = (L_sq - B2 * B2 - B3 * B3) / (2.0 * B2 * B3)
    cos_q3 = _clip_to_range(cos_q3, -1.0, 1.0)
    q3_candidates = [math.acos(cos_q3), -math.acos(cos_q3)]
    best = None
    best_err = float('inf')
    for q3 in q3_candidates:
        h = B3 * math.sin(q3)
        r_len = B2 + B3 * math.cos(q3)
        denom = L_sq
        if denom < 1e-12:
            continue
        cos_q2 = (x1 * h + z1 * r_len) / denom
        sin_q2 = (x1 * r_len - z1 * h) / denom
        cos_q2 = _clip_to_range(cos_q2, -1.0, 1.0)
        sin_q2 = _clip_to_range(sin_q2, -1.0, 1.0)
        q2 = math.atan2(sin_q2, cos_q2)
        q4 = theta_sum - q2 - q3
        for j in (q2, q3, q4):
            if not -2.0 * math.pi <= j <= 2.0 * math.pi:
                break
        else:
            x_chk = math.cos(q2) * h + math.sin(q2) * r_len
            z_chk = -math.sin(q2) * h + math.cos(q2) * r_len
            pos_err = math.hypot(x_chk - x1, z_chk - z1) + abs(y_error)
            orient_err = abs(q2 + q3 + q4 - theta_sum)
            err = pos_err + orient_err
            if err < best_err:
                best_err = err
                best = (q1, q2, q3, q4)
    return best