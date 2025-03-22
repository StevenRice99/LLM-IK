def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A purely analytical inverse kinematics solution for the specified 5-DOF manipulator,
    leveraging the same sub-chain approach as in "EXISTING 1" and "EXISTING 2," but
    carefully matching the geometry from the table:

        Joint 1: Revolute about Y, base at [0,0,0].
        Joint 2: Revolute about Y, at offset [0, -0.1197, 0.425] from Joint 1.
        Joint 3: Revolute about Y, at offset [0, 0, 0.39225] from Joint 2.
        Joint 4: Revolute about Z, at offset [0, 0.093, 0] from Joint 3.
        Joint 5: Revolute about Y, at offset [0, 0, 0.09465] from Joint 4.
        TCP:     offset [0, 0.0823, 0] and +90° about Z (1.570796325).

    Strategy:
      1) Solve Joint 1 (q1) from the top view: q1 = atan2(x, z).
      2) Transform the target (p, r) into the coordinate frame of Joint 2.
         That transform is T_base->joint1(q1) * Trans(0, -0.1197, 0.425).
         We invert that and apply it to (p, r).
      3) Solve the resulting 4-DOF sub-problem (joints q2..q5) about (Y, Y, Z, Y)
         using closed-form geometry. This follows the style of “EXISTING 2,”
         but with correct link lengths:
            - After Joint 2 (Y), an offset 0.39225 along Z (to Joint 3).
            - After Joint 3 (Y), an offset 0.093 along Y (to Joint 4).
            - After Joint 4 (Z), an offset 0.09465 along Z (to Joint 5).
            - After Joint 5 (Y), an offset 0.0823 along ? plus the final +90° about Z.
         We match orientation by searching over the ±arccos possibilities for the
         relevant angles and picking the candidate that best matches the target orientation.
      4) Return (q1, q2, q3, q4, q5) in radians.

    :param p: The desired TCP position [x, y, z].
    :param r: The desired TCP orientation [roll, pitch, yaw] in radians (URDF rpy).
    :return: A tuple (q1, q2, q3, q4, q5) that places the TCP at (p, r).
    """
    import math
    import numpy as np
    x, y, z = p
    roll, pitch, yaw = r
    q1 = math.atan2(x, z)

    def rot_y(a: float) -> np.ndarray:
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a: float) -> np.ndarray:
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

    def rot_x(a: float) -> np.ndarray:
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def trans_xyz(dx: float, dy: float, dz: float) -> np.ndarray:
        T = np.eye(4)
        T[0, 3], T[1, 3], T[2, 3] = (dx, dy, dz)
        return T

    def mat4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Convert rotation (3x3) and translation (3,) into homogeneous 4x4."""
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T

    def invert_homog(T: np.ndarray) -> np.ndarray:
        """Invert a 4x4 homogeneous transform."""
        R_ = T[0:3, 0:3]
        p_ = T[0:3, 3]
        R_inv = R_.T
        p_inv = -R_inv @ p_
        T_inv = np.eye(4)
        T_inv[0:3, 0:3] = R_inv
        T_inv[0:3, 3] = p_inv
        return T_inv
    Ry_q1 = rot_y(q1)
    Tb_j1 = mat4(Ry_q1, np.zeros(3))
    Tj1_j2 = trans_xyz(0.0, -0.1197, 0.425)
    Tb_j2 = Tb_j1 @ Tj1_j2
    Tj2_b = invert_homog(Tb_j2)
    R_tar = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    Tb_target = mat4(R_tar, np.array([x, y, z]))
    T2_target = Tj2_b @ Tb_target
    p2 = T2_target[0:3, 3]
    R2 = T2_target[0:3, 0:3]

    def rpy_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
        beta = math.asin(-R[2, 0])
        alpha = math.atan2(R[2, 1], R[2, 2])
        gamma = math.atan2(R[1, 0], R[0, 0])
        return (alpha, beta, gamma)
    roll_sub, pitch_sub, yaw_sub = rpy_from_matrix(R2)

    def solve_sub4(p_sub: np.ndarray, rpy_sub: tuple[float, float, float]) -> tuple[float, float, float, float]:
        """
        Returns (q2, q3, q4, q5) for the sub-chain, using a direct geometry approach.
        """
        d23 = 0.39225
        d34y = 0.093
        d45 = 0.09465
        d_tcp = 0.0823
        psi = 1.570796325
        x_s, y_s, z_s = p_sub
        r_sub, p_sub_, y_sub = rpy_sub

        def Rx(a):
            return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

        def Ry(a):
            return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

        def Rz(a):
            return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        R_sub_target = Rz(y_sub) @ Ry(p_sub_) @ Rx(r_sub)
        cos_q3 = (y_s - d34y) / d_tcp
        cos_q3 = max(-1.0, min(1.0, cos_q3))
        q3_candidates = [math.acos(cos_q3), -math.acos(cos_q3)]
        r_xy = math.sqrt(x_s ** 2 + z_s ** 2)
        delta = math.atan2(x_s, z_s)
        best_err = 1000000000.0
        best_sol = (0, 0, 0, 0)
        for q3_ in q3_candidates:
            s3 = math.sin(q3_)
            L34 = math.sqrt(d45 ** 2 + (d_tcp * s3) ** 2)
            phi = math.atan2(d_tcp * s3, d45)
            cos_term = (r_xy ** 2 + d23 ** 2 - L34 ** 2) / (2 * d23 * r_xy)
            cos_term = max(-1.0, min(1.0, cos_term))
            alpha_offset = math.acos(cos_term)
            for signA in [1, -1]:
                q2_ = delta + signA * alpha_offset
                Vx = x_s - d23 * math.sin(q2_)
                Vz = z_s - d23 * math.cos(q2_)
                q2_3 = math.atan2(Vx, Vz) + phi
                q3_sub = q2_3 - q2_
                R_pre = Ry(q2_ + q3_sub) @ Rz(0)
                if abs(q3_sub - q3_) > 0.01:
                    continue

                def Ry(a):
                    ca, sa = (math.cos(a), math.sin(a))
                    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

                def Rz(a):
                    ca, sa = (math.cos(a), math.sin(a))
                    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
                R_pre = Ry(q2_ + q3_)
                R_needed = R_pre.T @ R_sub_target @ Rz(-1.570796325)
                alpha_4 = math.atan2(R_needed[1, 0], R_needed[0, 0])
                R_temp = Rz(-alpha_4) @ R_needed
                beta_5 = math.atan2(R_temp[0, 2], R_temp[0, 0])
                R_check = R_pre @ Rz(alpha_4) @ Ry(beta_5) @ Rz(1.570796325)
                diff = R_check - R_sub_target
                err_ = np.linalg.norm(diff, 'fro')
                if err_ < best_err:
                    best_err = err_
                    best_sol = (q2_, q3_, alpha_4, beta_5)
        return best_sol
    q2_, q3_, q4_, q5_ = solve_sub4(p2, (roll_sub, pitch_sub, yaw_sub))
    return (q1, q2_, q3_, q4_, q5_)