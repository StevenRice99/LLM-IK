import math
from typing import Tuple
L = 0.425
B = 0.01615

def _wrap(angle: float) -> float:
    """wrap any angle to the interval [−π , π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def _ang_diff(a: float, b: float) -> float:
    """shortest signed difference a−b (mod 2π)."""
    return _wrap(a - b)

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Closed‑form IK for the 2‑DoF serial manipulator.

    Parameters
    ----------
    p : (x, y, z)
        Desired TCP position (metres) in the base frame.
    r : (roll, pitch, yaw)
        Desired TCP orientation in URDF XYZ‑RPY convention (radians).

    Returns
    -------
    (θ1, θ2) : Tuple[float, float]
        Joint angles in radians that realise the requested pose.
    """
    x_d, y_d, z_d = p
    roll_d, pitch_d, yaw_d = r
    cos_t2 = max(-1.0, min(1.0, z_d / L))
    abs_t2 = math.acos(cos_t2)
    candidates = []
    for theta2 in (abs_t2, -abs_t2):
        sin_t2 = math.sin(theta2)
        A = L * sin_t2
        denom = A * A + B * B
        C = (A * x_d + B * y_d) / denom
        S = (-B * x_d + A * y_d) / denom
        theta1 = math.atan2(S, C)
        if cos_t2 >= 0.0:
            roll_p = 0.0
            pitch_p = theta2
            yaw_p = _wrap(theta1)
        else:
            roll_p = math.pi
            pitch_p = _wrap(math.pi - theta2)
            yaw_p = _wrap(theta1 + math.pi)
        candidates.append((theta1, theta2, roll_p, pitch_p, yaw_p))
    best, best_err = (None, float('inf'))
    for th1, th2, r_p, p_p, y_p in candidates:
        err = abs(_ang_diff(r_p, roll_d)) + abs(_ang_diff(p_p, pitch_d)) + abs(_ang_diff(y_p, yaw_d))
        if err < best_err:
            best_err = err
            best = (th1, th2)
    theta1_final = _wrap(best[0])
    theta2_final = _wrap(best[1])
    return (theta1_final, theta2_final)