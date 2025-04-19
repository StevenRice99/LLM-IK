import math
from typing import Tuple
D_Y1 = 0.13585
D_Y2 = -0.1197
Y_OFF = D_Y1 + D_Y2
L1 = 0.425
L2 = 0.39225

def _Rz(theta: float) -> Tuple[Tuple[float, ...], ...]:
    c, s = (math.cos(theta), math.sin(theta))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))

def _Ry(theta: float) -> Tuple[Tuple[float, ...], ...]:
    c, s = (math.cos(theta), math.sin(theta))
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))

def _rpy_to_matrix(rx: float, ry: float, rz: float) -> Tuple[Tuple[float, ...], ...]:
    cx, sx = (math.cos(rx), math.sin(rx))
    cy, sy = (math.cos(ry), math.sin(ry))
    cz, sz = (math.cos(rz), math.sin(rz))
    return ((cy * cz, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx), (cy * sz, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx), (-sy, cy * sx, cy * cx))

def _mat_diff(A, B) -> float:
    """Frobenius‑norm difference ‖A − B‖_F."""
    return sum(((A[i][j] - B[i][j]) ** 2 for i in range(3) for j in range(3)))

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Returns the joint angles θ₁, θ₂, θ₃ (radians) that bring the TCP to the
    requested position *p* and orientation *r* (roll‑pitch‑yaw, URDF rule).
    Reachability is guaranteed by the task statement.
    """
    x, y, z = p
    rho = math.hypot(x, y)
    if rho < 1e-12:
        rho = 1e-12
    gamma = math.atan2(y, x)
    delta = -Y_OFF / rho
    delta = max(min(delta, 1.0), -1.0)
    theta1_opts = (gamma + math.asin(delta), gamma + math.pi - math.asin(delta))
    rx, ry, rz = r
    R_des = _rpy_to_matrix(rx, ry, rz)
    best_err = float('inf')
    best_trip = (0.0, 0.0, 0.0)
    for theta1 in theta1_opts:
        c1, s1 = (math.cos(theta1), math.sin(theta1))
        x1 = x * c1 + y * s1
        y1 = -x * s1 + y * c1
        z1 = z
        if abs(y1 - Y_OFF) > 5e-06:
            continue
        xp = x1
        zp = z1
        d2 = xp * xp + zp * zp
        c3 = (d2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        c3 = max(min(c3, 1.0), -1.0)
        for theta3 in (math.acos(c3), -math.acos(c3)):
            s3 = math.sin(theta3)
            k1 = L1 + L2 * c3
            k2 = L2 * s3
            theta2 = math.atan2(xp, zp) - math.atan2(k2, k1)
            R_sol = _Rz(theta1)
            Ry2 = _Ry(theta2)
            Ry3 = _Ry(theta3)
            R_tmp = tuple((tuple((sum((R_sol[i][k] * Ry2[k][j] for k in range(3))) for j in range(3))) for i in range(3)))
            R_sol = tuple((tuple((sum((R_tmp[i][k] * Ry3[k][j] for k in range(3))) for j in range(3))) for i in range(3)))
            err = _mat_diff(R_sol, R_des)
            if err < best_err:
                best_err = err
                best_trip = (theta1, theta2, theta3)
                if err < 1e-06:
                    return best_trip
    return best_trip