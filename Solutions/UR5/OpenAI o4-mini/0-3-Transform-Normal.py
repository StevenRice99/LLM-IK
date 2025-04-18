import math
from typing import Tuple

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """
    Closed‑form IK for the 4‑DOF Rz–Ry–Ry–Ry arm with a TCP offset [0,0.093,0].
    p = (px,py,pz): desired TCP position in base frame.
    r = (roll,pitch,yaw): desired extrinsic RPY (roll≈0 or ≈±π).
    Returns (theta1, theta2, theta3, theta4) in radians.
    """
    px, py, pz = p
    roll, pitch, yaw = r

    def _wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi
    C0 = 0.13585 - 0.1197 + 0.093
    rho = math.hypot(px, py)
    if rho < 1e-08:
        raise ValueError('Singular or unreachable: px,py too small')
    δ = math.atan2(px, py)
    arg = C0 / rho
    arg = max(min(arg, 1.0), -1.0)
    Ψ = math.acos(arg)
    t1a = _wrap(-δ + Ψ)
    t1b = _wrap(-δ - Ψ)
    if math.cos(roll) >= 0.0:
        yaw_ref = yaw
    else:
        yaw_ref = yaw + math.pi
    if abs(_wrap(t1a - yaw_ref)) < abs(_wrap(t1b - yaw_ref)):
        theta1 = t1a
    else:
        theta1 = t1b
    x2 = math.cos(theta1) * px + math.sin(theta1) * py
    z2 = pz
    L2 = 0.425
    L3 = 0.39225
    D = (x2 * x2 + z2 * z2 - L2 * L2 - L3 * L3) / (2 * L2 * L3)
    D = max(min(D, 1.0), -1.0)
    s3 = math.sqrt(max(0.0, 1.0 - D * D))
    theta3 = math.atan2(s3, D)
    K1 = L2 + L3 * math.cos(theta3)
    K2 = L3 * s3
    theta2 = math.atan2(K1 * x2 - K2 * z2, K1 * z2 + K2 * x2)
    if math.cos(roll) >= 0.0:
        φ = pitch
    else:
        φ = math.pi - pitch
    theta4 = φ - (theta2 + theta3)
    return (_wrap(theta1), _wrap(theta2), _wrap(theta3), _wrap(theta4))