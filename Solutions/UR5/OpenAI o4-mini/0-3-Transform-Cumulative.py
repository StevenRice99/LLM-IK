import math
import numpy as np

def _rotz(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

def _roty(a: float) -> np.ndarray:
    ca, sa = (math.cos(a), math.sin(a))
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

def _rpy_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """
    Extract RPY from R = Rz(yaw)·Ry(pitch)·Rx(roll).
    Returns (roll, pitch, yaw), each in (-π,π].
    """
    sy = -R[2, 0]
    cy = math.hypot(R[0, 0], R[1, 0])
    pitch = math.atan2(sy, cy)
    if cy < 1e-06:
        roll = 0.0
        yaw = math.atan2(R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])

    def _wrap(x):
        return (x + math.pi) % (2 * math.pi) - math.pi
    return (_wrap(roll), _wrap(pitch), _wrap(yaw))

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    :param p: target TCP pos [x,y,z]
    :param r: target TCP rpy [roll, pitch, yaw]
    :return: (theta1, theta2, theta3, theta4)
    """
    px, py, pz = p
    roll_t, pitch_t, yaw_t = r
    cr, sr = (math.cos(roll_t), math.sin(roll_t))
    cp, sp = (math.cos(pitch_t), math.sin(pitch_t))
    cy, sy = (math.cos(yaw_t), math.sin(yaw_t))
    R0_e = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    theta1 = math.atan2(-R0_e[0, 1], R0_e[1, 1])
    R1_e = _rotz(-theta1) @ R0_e
    phi = math.atan2(R1_e[0, 2], R1_e[0, 0])
    d_tcp = 0.093
    d_off = d_tcp * R0_e[:, 1]
    Pw = np.array([px, py, pz]) - d_off
    P1 = _rotz(-theta1) @ Pw
    x2, z2 = (P1[0], P1[2])
    L1, L2 = (0.425, 0.39225)
    d2 = x2 * x2 + z2 * z2
    cos3 = (d2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos3 = max(min(cos3, 1.0), -1.0)
    sol_candidates = []
    for t3 in (math.acos(cos3), -math.acos(cos3)):
        C = L1 + L2 * math.cos(t3)
        D = L2 * math.sin(t3)
        denom = C * C + D * D
        if denom < 1e-08:
            continue
        sin2 = (C * x2 - D * z2) / denom
        cos2 = (D * x2 + C * z2) / denom
        sin2 = max(min(sin2, 1.0), -1.0)
        cos2 = max(min(cos2, 1.0), -1.0)
        t2 = math.atan2(sin2, cos2)
        t4 = phi - t2 - t3
        t4 = (t4 + math.pi) % (2 * math.pi) - math.pi
        sol_candidates.append((theta1, t2, t3, t4))
    best = None
    best_err = 1000000000.0
    for t1, t2, t3, t4 in sol_candidates:
        R04 = _rotz(t1) @ _roty(t2) @ _roty(t3) @ _roty(t4)
        rsol = _rpy_from_R(R04)
        err = 0.0
        for a, b in zip(rsol, (roll_t, pitch_t, yaw_t)):
            diff = abs(a - b) % (2 * math.pi)
            err += min(diff, 2 * math.pi - diff)
        if err < best_err:
            best_err = err
            best = (t1, t2, t3, t4)
    return best if best is not None else sol_candidates[0]