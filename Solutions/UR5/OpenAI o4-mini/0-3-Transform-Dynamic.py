import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    4‑DOF IK for Z–Y–Y–Y arm.  Tries both base‐angle roots and both elbow‐up/down,
    reconstructs each candidate’s full TCP RPY, and picks the one minimizing
    the sum of RPY errors (with a tiny q2‐tie‐breaker).
    """
    x, y, z = p
    roll_d, pitch_d, yaw_d = r
    d2_y, d3_y = (0.13585, -0.1197)
    L1 = 0.425
    L2 = 0.39225
    dtcp_y = 0.093
    y_const = d2_y + d3_y + dtcp_y
    cr, sr = (math.cos(roll_d), math.sin(roll_d))
    cp, sp = (math.cos(pitch_d), math.sin(pitch_d))
    cy, sy = (math.cos(yaw_d), math.sin(yaw_d))
    R_des = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    TWO_PI = 2.0 * math.pi

    def normalize_ang(a: float) -> float:
        """Wrap into (–π, π]"""
        return (a + math.pi) % TWO_PI - math.pi

    def rot_z(a: float) -> np.ndarray:
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def rot_y(a: float) -> np.ndarray:
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def ang_diff(a: float, b: float) -> float:
        """minimal signed difference a–b in (–π,π]"""
        d = (a - b + math.pi) % TWO_PI - math.pi
        return d
    best_cost = float('inf')
    best_sol = (0.0, 0.0, 0.0, 0.0)
    rho = math.hypot(x, y)
    phi = math.atan2(-x, y)
    alpha = math.acos(y_const / rho)
    q1_cands = [phi - alpha, phi + alpha]
    for q1 in q1_cands:
        c1, s1 = (math.cos(q1), math.sin(q1))
        x1 = c1 * x + s1 * y
        z1 = z
        cos_q3 = (x1 * x1 + z1 * z1 - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        cos_q3 = max(min(cos_q3, 1.0), -1.0)
        for q3 in (math.acos(cos_q3), -math.acos(cos_q3)):
            A = L1 + L2 * math.cos(q3)
            B = L2 * math.sin(q3)
            D = A * A + B * B
            if D < 1e-09:
                continue
            sin_q2 = (A * x1 - B * z1) / D
            cos_q2 = (A * z1 + B * x1) / D
            sin_q2 = max(min(sin_q2, 1.0), -1.0)
            cos_q2 = max(min(cos_q2, 1.0), -1.0)
            q2 = math.atan2(sin_q2, cos_q2)
            R1_inv = rot_z(-q1) @ R_des
            target_sum = math.atan2(R1_inv[0, 2], R1_inv[0, 0])
            q4 = target_sum - (q2 + q3)
            q1n = normalize_ang(q1)
            q2n = normalize_ang(q2)
            q3n = normalize_ang(q3)
            q4n = normalize_ang(q4)
            Qsum = q2n + q3n + q4n
            R_end = rot_z(2 * q1n) @ rot_y(Qsum) @ rot_z(-q1n)
            pitch_e = math.atan2(-R_end[2, 0], math.hypot(R_end[0, 0], R_end[1, 0]))
            roll_e = math.atan2(R_end[2, 1], R_end[2, 2])
            yaw_e = math.atan2(R_end[1, 0], R_end[0, 0])
            err = abs(ang_diff(roll_e, roll_d)) + abs(ang_diff(pitch_e, pitch_d)) + abs(ang_diff(yaw_e, yaw_d))
            cost = err + 0.001 * abs(q2n)
            if cost < best_cost:
                best_cost = cost
                best_sol = (q1n, q2n, q3n, q4n)
    return best_sol