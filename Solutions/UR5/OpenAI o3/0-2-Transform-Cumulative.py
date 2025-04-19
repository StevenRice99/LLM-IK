import math
from typing import Tuple
import numpy as np

def _rotx(a: float) -> np.ndarray:
    c, s = (math.cos(a), math.sin(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

def _roty(a: float) -> np.ndarray:
    c, s = (math.cos(a), math.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def _rotz(a: float) -> np.ndarray:
    c, s = (math.cos(a), math.sin(a))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Analytical inverse kinematics for the 3‑DOF manipulator.

    Parameters
    ----------
    p : (x, y, z)
        Desired TCP position in metres, expressed in the base frame.
    r : (roll, pitch, yaw)
        Desired TCP orientation as URDF‐style RPY angles (rad).

    Returns
    -------
    (θ1, θ2, θ3) : Tuple[float, float, float]
        Joint angles (rad) that place the TCP at the requested pose.
    """
    d1 = 0.13585
    t23 = np.array([0.0, -0.1197, 0.425])
    tTCP = np.array([0.0, 0.0, 0.39225])
    rx, ry, rz = r
    R = _rotz(rz) @ _roty(ry) @ _rotx(rx)
    theta1 = math.atan2(-R[0, 1], R[1, 1])
    p_tcp = np.asarray(p, dtype=float)
    p3 = p_tcp - R @ tTCP
    p1 = np.array([-d1 * math.sin(theta1), d1 * math.cos(theta1), 0.0])
    delta = p3 - p1
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    RzT = np.array([[c1, s1, 0.0], [-s1, c1, 0.0], [0.0, 0.0, 1.0]])
    v23 = RzT @ delta
    theta2 = math.atan2(v23[0], v23[2])
    theta23 = math.atan2(-R[2, 0], R[2, 2])
    theta3 = theta23 - theta2
    return (theta1, theta2, theta3)