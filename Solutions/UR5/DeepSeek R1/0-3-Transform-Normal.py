import numpy as np
from math import atan2, acos, sin, cos, sqrt, pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    px, py, pz = p
    rx, ry, rz = r
    q1_1 = atan2(py, px)
    q1_2 = (q1_1 + pi) % (2 * pi)
    for q1 in [q1_1, q1_2]:
        link2_x = 0.13585 * sin(q1)
        link2_y = 0.13585 * cos(q1)
        adj_x = px - link2_x
        adj_y = py - link2_y
        adj_z = pz
        A = sqrt(adj_x ** 2 + adj_y ** 2)
        D = sqrt(A ** 2 + adj_z ** 2)
        L1 = 0.425
        L2 = 0.39225 + 0.093
        if not abs(L1 - L2) - 1e-06 <= D <= L1 + L2 + 1e-06:
            continue
        cos_q3 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_q3 = np.clip(cos_q3, -1.0, 1.0)
        q3 = acos(cos_q3)
        for q3_sign in [1, -1]:
            q3_val = q3 * q3_sign
            alpha = atan2(L2 * sin(q3_val), L1 + L2 * cos(q3_val))
            q2 = atan2(A, adj_z) - alpha
            crx, srx = (cos(rx), sin(rx))
            cry, sry = (cos(ry), sin(ry))
            crz, srz = (cos(rz), sin(rz))
            R_rx = np.array([[1, 0, 0], [0, crx, -srx], [0, srx, crx]])
            R_ry = np.array([[cry, 0, sry], [0, 1, 0], [-sry, 0, cry]])
            R_rz = np.array([[crz, -srz, 0], [srz, crz, 0], [0, 0, 1]])
            R_desired = R_rz @ R_ry @ R_rx
            R_z_inv = np.array([[cos(-q1), -sin(-q1), 0], [sin(-q1), cos(-q1), 0], [0, 0, 1]])
            R_rot = R_z_inv @ R_desired
            theta = atan2(R_rot[0, 2], R_rot[0, 0])
            q4 = (theta - q2 - q3_val) % (2 * pi)
            if q4 > pi:
                q4 -= 2 * pi
            return (q1, q2, q3_val, q4)
    return (q1_1, 0.0, 0.0, 0.0)