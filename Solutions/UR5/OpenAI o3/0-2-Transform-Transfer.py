import math
from typing import Tuple, List

def _wrap(a: float) -> float:
    while a <= -math.pi:
        a += 2.0 * math.pi
    while a > math.pi:
        a -= 2.0 * math.pi
    return a

def _rotz(t: float):
    c, s = (math.cos(t), math.sin(t))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

def _roty(t: float):
    c, s = (math.cos(t), math.sin(t))
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))

def _rotx(t: float):
    c, s = (math.cos(t), math.sin(t))
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))

def _mat_mul(a, b):
    return tuple((tuple((sum((a[i][k] * b[k][j] for k in range(3))) for j in range(3))) for i in range(3)))

def _solve_position(x: float, y: float, z: float) -> List[Tuple[float, float, float]]:
    """
    Enumerate every (θ1, θ2, θ3) that reproduces the requested position.
    Up to four solutions exist (elbow‑up / elbow‑down and mirror symmetry).
    """
    k = 0.01615
    a2 = 0.425
    a3 = 0.39225
    k2 = k * k
    num = x * x + y * y + z * z - (a2 * a2 + a3 * a3 + k2)
    den = 2.0 * a2 * a3
    c3 = max(min(num / den, 1.0), -1.0)
    θ3_roots = [math.acos(c3), -math.acos(c3)]
    sols: List[Tuple[float, float, float]] = []
    for θ3 in θ3_roots:
        C = a2 + a3 * math.cos(θ3)
        D = a3 * math.sin(θ3)
        A2 = x * x + y * y - k2
        A_abs = math.sqrt(max(A2, 0.0))
        for sign in (+1.0, -1.0):
            A = sign * A_abs
            sin2 = (C * A - D * z) / (C * C + D * D)
            cos2 = (C * z + D * A) / (C * C + D * D)
            θ2 = math.atan2(sin2, cos2)
            if x == 0.0 and y == 0.0:
                θ1 = 0.0
            else:
                sin1_num = A * y - k * x
                cos1_num = A * x + k * y
                θ1 = math.atan2(sin1_num, cos1_num)
            sols.append((_wrap(θ1), _wrap(θ2), _wrap(θ3)))
    return sols

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Closed‑form inverse kinematics for the 3‑DoF chain
           Revolute‑Z  →  Revolute‑Y  →  Revolute‑Y  →  TCP
    Parameters
    ----------
    p : (x, y, z)   –  TCP position in metres.
    r : (roll, pitch, yaw)   – desired fixed‑axis XYZ (rpy) orientation.
    Returns
    -------
    (θ1, θ2, θ3)  –  joint angles in radians, wrapped to (‑π, π].
    """
    roll_d, pitch_d, yaw_d = r
    Rdes = _mat_mul(_mat_mul(_rotz(yaw_d), _roty(pitch_d)), _rotx(roll_d))
    n11, n12, n13 = Rdes[0]
    n21, n22, n23 = Rdes[1]
    n31, n32, n33 = Rdes[2]
    θ1_d = math.atan2(-n12, n22)
    c1 = math.cos(θ1_d)
    s1 = math.sin(θ1_d)
    if abs(c1) > 1e-08:
        s23 = n13 / c1
    else:
        s23 = n23 / s1
    c23 = n33
    θ23_d = math.atan2(s23, c23)
    θ1_d = _wrap(θ1_d)
    θ23_d = _wrap(θ23_d)
    best_set = None
    best_err = float('inf')
    for θ1, θ2, θ3 in _solve_position(*p):
        err = abs(_wrap(θ1 - θ1_d)) + abs(_wrap(θ2 + θ3 - θ23_d))
        if err < best_err:
            best_err = err
            best_set = (θ1, θ2, θ3)
            if err < 0.0001:
                break
    assert best_set is not None, 'No feasible IK solution found.'
    return best_set