import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Inverse kinematics for a 5-DOF manipulator with:
      - Revolute 1 at base (origin, axis Y)
      - Revolute 2 offset by [0, -0.1197, 0.425] from base (axis Y)
      - Revolute 3 with translation [0, 0, 0.39225] (axis Y)
      - Revolute 4 with translation [0, 0.093, 0] (axis Z)
      - Revolute 5 with translation [0, 0, 0.09465] (axis Y)
      - TCP with translation [0, 0.0823, 0] and fixed rotation about Z by psi=1.570796325.
      
    The solution decouples the position and orientation by “removing” the TCP offset.
    (Note: This method generates several candidate solutions and selects the one with the smallest
     orientation error.)
    
    Parameters:
      p: Desired TCP position in global coordinates [x, y, z].
      r: Desired TCP orientation in roll, pitch, yaw (radians) (URDF convention: R_target = Rz(yaw) @ Ry(pitch) @ Rx(roll)).
      
    Returns:
      A tuple (q1, q2, q3, q4, q5) of joint angles in radians.
    """

    def rot_x(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    off_12 = np.array([0.0, -0.1197, 0.425])
    d2 = 0.39225
    d3 = 0.093
    d4 = 0.09465
    d_tcp = 0.0823
    psi = 1.570796325
    roll, pitch, yaw = r
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    p_global = np.array(p)
    p_wrist = p_global - R_target @ np.array([0, d_tcp, 0])
    q1 = math.atan2(p_wrist[0], p_wrist[2])
    R_y_q1 = rot_y(q1)
    p_joint2 = R_y_q1 @ off_12
    p_sub = rot_y(-q1) @ (p_wrist - p_joint2)
    px, py, pz = p_sub
    r_planar = math.sqrt(px ** 2 + pz ** 2)
    delta = math.atan2(px, pz)
    cos_angle = (r_planar ** 2 - d2 ** 2 - d4 ** 2) / (2 * d2 * d4)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    alpha = math.acos(cos_angle)
    phi = math.atan2(d4 * math.sin(alpha), d2 + d4 * math.cos(alpha))
    candidate_sets = []
    q2_A = delta - phi
    q23_A = delta + phi
    candidate_sets.append((q2_A, q23_A))
    q2_B = delta + phi
    q23_B = delta - phi
    candidate_sets.append((q2_B, q23_B))
    candidates = []
    for q2_candidate, q23 in candidate_sets:
        R_target_wrist = rot_y(-q23) @ rot_y(-q1) @ R_target @ rot_z(-psi)
        val = R_target_wrist[0, 1]
        val = max(min(val, 1.0), -1.0)
        q4_cand_1 = -math.asin(val)
        q4_cand_2 = math.pi - q4_cand_1
        for q4_candidate in [q4_cand_1, q4_cand_2]:
            q3_candidate = q23 - q2_candidate - q4_candidate
            R_pre = rot_y(q1) @ rot_y(q2_candidate + q3_candidate) @ rot_z(q4_candidate)
            R_wrist = np.linalg.inv(R_pre) @ R_target @ rot_z(-psi)
            q5_candidate = math.atan2(R_wrist[0, 2], R_wrist[0, 0])
            R_candidate = rot_y(q1) @ rot_y(q2_candidate + q3_candidate) @ rot_z(q4_candidate) @ rot_y(q5_candidate) @ rot_z(psi)
            error = np.linalg.norm(R_candidate - R_target, ord='fro')
            candidates.append((error, (q1, q2_candidate, q3_candidate, q4_candidate, q5_candidate)))
    best_candidate = min(candidates, key=lambda c: c[0])[1]
    return best_candidate