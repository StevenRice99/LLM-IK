import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    4‐DOF closed‐form IK with robust branch selection by
    true rotation‐angle error for R_04 vs. R_target.
    """
    x, y, z = p
    roll, pitch, yaw = r
    d1, d2y, d2z, d3z, d4y = (0.13585, -0.1197, 0.425, 0.39225, 0.093)
    u_y = d1 + d2y + d4y

    def Rz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def Ry(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rx(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    R_target = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    E = 2.0 * d2z * d3z
    D = d2z * d2z + d3z * d3z
    cos_q3 = (x * x + y * y + z * z - u_y * u_y - D) / E
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3_options = [np.arccos(cos_q3), -np.arccos(cos_q3)]
    best_err = np.inf
    best_solution = (0.0, 0.0, 0.0, 0.0)
    for q3 in q3_options:
        A = d2z + d3z * np.cos(q3)
        B = d3z * np.sin(q3)
        C = np.hypot(A, B)
        gamma = np.arctan2(B, A)
        phi = np.arccos(np.clip(z / C, -1.0, 1.0))
        q2_candidates = [-gamma + phi, -gamma - phi]
        for q2 in q2_candidates:
            q2n = (q2 + np.pi) % (2 * np.pi) - np.pi
            u_x = B * np.cos(q2n) + A * np.sin(q2n)
            phi_off = np.arctan2(u_y, u_x)
            q1 = np.arctan2(y, x) - phi_off
            q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
            R_03 = Rz(q1) @ Ry(q2n) @ Ry(q3)
            R_diff = R_03.T @ R_target
            q4 = np.arctan2(R_diff[0, 2], R_diff[2, 2])
            q4 = (q4 + np.pi) % (2 * np.pi) - np.pi
            R_04 = R_03 @ Ry(q4)
            R_err = R_04.T @ R_target
            cos_err = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
            angle_err = abs(np.arccos(cos_err))
            if angle_err < best_err:
                best_err = angle_err
                best_solution = (q1, q2n, q3, q4)
    return best_solution