import math
from typing import Tuple
D1 = 0.13585
D2 = -0.1197
L = 0.425
D = D1 + D2

def _wrap(a: float) -> float:
    """map angle to the interval [−π , π]"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def _tool_z(theta1: float, theta2: float) -> Tuple[float, float, float]:
    """Direction of the tool Z‑axis in the base frame."""
    s2, c2 = (math.sin(theta2), math.cos(theta2))
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    return (s2 * c1, s2 * s1, c2)

def _desired_z(r: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Requested tool Z‑axis from the desired roll‑pitch‑yaw triple."""
    roll, pitch, yaw = r
    cy, sy = (math.cos(yaw), math.sin(yaw))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cr, sr = (math.cos(roll), math.sin(roll))
    zx = cy * sp * cr + sy * sr
    zy = sy * sp * cr - cy * sr
    zz = cp * cr
    n = math.hypot(math.hypot(zx, zy), zz)
    return (zx / n, zy / n, zz / n)

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Closed‑form inverse kinematics for the 2‑DOF chain.

    Parameters
    ----------
    p : (x, y, z)   desired TCP position in the base frame  [m]
    r : (roll, pitch, yaw)   desired TCP orientation in RPY [rad]

    Returns
    -------
    (theta1, theta2)   joint angles in radians, each wrapped to [−π ,\xa0π].
    """
    x, y, z = p
    c2 = max(-1.0, min(1.0, z / L))
    theta2_candidates = [math.acos(c2), -math.acos(c2)]
    solutions: list[Tuple[float, float]] = []
    for theta2 in theta2_candidates:
        s2 = math.sin(theta2)
        ls2 = L * s2
        if abs(ls2) < 1e-12:
            theta1 = math.atan2(-x, y)
        else:
            theta1 = math.atan2(y, x) - math.atan2(D, ls2)
        solutions.append((_wrap(theta1), _wrap(theta2)))
    zx_d, zy_d, zz_d = _desired_z(r)
    best: Tuple[float, float] = solutions[0]
    best_dot = -2.0
    for theta1, theta2 in solutions:
        zx, zy, zz = _tool_z(theta1, theta2)
        dot = zx * zx_d + zy * zy_d + zz * zz_d
        if dot > best_dot:
            best_dot = dot
            best = (theta1, theta2)
    return best