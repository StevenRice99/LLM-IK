import math
from typing import Tuple

def inverse_kinematics(p: Tuple[float, float, float]) -> Tuple[float, float, float, float, float]:
    """
    Closed‐form IK for the 5‐DOF arm (position only).
    We lock q4=q5=0 and solve q1,q2,q3 so that the TCP at:
      T01(q1)·T12(T2)·T23(T3)·T34(T4)·T45(T5)·T5T
    reaches p.  Here we absorb the last two links into
    an equivalent 2–link planar problem.
    """
    x, y, z = p
    d2 = 0.13585
    b1 = 0.1197
    c1 = 0.425
    d4 = 0.39225
    b3 = 0.093
    d6 = 0.09465
    e = d2 - b1 + b3
    r = math.hypot(x, y)
    M = math.sqrt(max(0.0, r * r - e * e))
    alpha = math.atan2(y, x)
    beta = math.atan2(e, M)
    q1 = alpha - beta
    L1 = c1
    L2 = d4 + d6
    D = (M * M + z * z - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    D = max(-1.0, min(1.0, D))
    q3 = math.acos(D)
    phi = math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    q2 = math.atan2(z, M) - phi
    q4 = 0.0
    q5 = 0.0
    return (q1, q2, q3, q4, q5)