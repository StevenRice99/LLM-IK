import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r"
    using a Jacobian-based iterative method (standard numerical IK).
    
    We do not use any optimization library calls; this is a direct Newton-like
    approach for solving the 6D pose error (3 for position, 3 for orientation)
    with the 4 unknowns, by using a pseudo-inverse of the 6×4 Jacobian.
    
    :param p: The desired position in the form [x, y, z].
    :param r: The desired orientation in radians [roll, pitch, yaw].
    :return: A 4-tuple of joint angles [joint1, joint2, joint3, joint4] in radians.
    """

    def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward kinematics given joint angles q = [q1, q2, q3, q4].
        Returns (pos, R) where pos is 3×1, R is 3×3.
        
        Joint layout (from the DETAILS):
          Joint1: revolute about Z, origin at base
          Joint2: revolute about Y, offset [0, 0.13585, 0]
          Joint3: revolute about Y, offset [0, -0.1197, 0.425]
          Joint4: revolute about Y, offset [0, 0, 0.39225]
          TCP offset: [0, 0, 0.093]
        """
        q1, q2, q3, q4 = q

        def rotZ(th):
            return np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]], dtype=float)

        def rotY(th):
            return np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]], dtype=float)

        def trans(tx, ty, tz):
            T = np.eye(4, dtype=float)
            T[0, 3] = tx
            T[1, 3] = ty
            T[2, 3] = tz
            return T

        def rot_to_4x4(R):
            T = np.eye(4, dtype=float)
            T[0:3, 0:3] = R
            return T
        T0_1 = rot_to_4x4(rotZ(q1))
        T1_2 = trans(0, 0.13585, 0) @ rot_to_4x4(rotY(q2))
        T2_3 = trans(0, -0.1197, 0.425) @ rot_to_4x4(rotY(q3))
        T3_4 = trans(0, 0, 0.39225) @ rot_to_4x4(rotY(q4))
        T4_TCP = trans(0, 0, 0.093)
        T0_4 = T0_1 @ T1_2 @ T2_3 @ T3_4
        T0_TCP = T0_4 @ T4_TCP
        pos = T0_TCP[0:3, 3].copy()
        R_ee = T0_TCP[0:3, 0:3].copy()
        return (pos, R_ee)

    def orientation_error(R_des: np.ndarray, R_ee: np.ndarray) -> np.ndarray:
        R_err = R_des.T @ R_ee
        rx = 0.5 * (R_err[2, 1] - R_err[1, 2])
        ry = 0.5 * (R_err[0, 2] - R_err[2, 0])
        rz = 0.5 * (R_err[1, 0] - R_err[0, 1])
        return np.array([rx, ry, rz], dtype=float)
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp_ = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_des = np.array([[cy * cp, cy * sp_ * sr - sy * cr, cy * sp_ * cr + sy * sr], [sy * cp, sy * sp_ * sr + cy * cr, sy * sp_ * cr - cy * sr], [-sp_, cp * sr, cp * cr]], dtype=float)
    target_pos = np.array(p, dtype=float)

    def error_func(q: np.ndarray) -> np.ndarray:
        pos_fk, R_fk = fk(q)
        pos_err = pos_fk - target_pos
        ori_err = orientation_error(R_des, R_fk)
        return np.concatenate([pos_err, ori_err])

    def jacobian(q: np.ndarray) -> np.ndarray:
        eps = 1e-06
        base_err = error_func(q)
        J = np.zeros((6, 4), dtype=float)
        for i in range(4):
            dq = np.zeros(4, dtype=float)
            dq[i] = eps
            err_plus = error_func(q + dq)
            J[:, i] = (err_plus - base_err) / eps
        return J
    q = np.zeros(4, dtype=float)
    for _iter in range(50):
        e_val = error_func(q)
        norm_e = np.linalg.norm(e_val)
        if norm_e < 1e-06:
            break
        J = jacobian(q)
        JTJ = J.T @ J
        JTJ += 1e-09 * np.eye(4)
        dq = np.linalg.inv(JTJ) @ J.T @ -e_val
        q += dq
    for i in range(4):
        q[i] = (q[i] + math.pi) % (2 * math.pi) - math.pi
    return tuple(q)