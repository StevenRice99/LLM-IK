import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Computes the joint angles for a 4-DOF serial manipulator that reaches a specified
    TCP position and orientation using a closed-form analytical inverse kinematics solution.
    
    The robot structure (all units in meters and radians):
      - Revolute Joint 1: rotation about Y, at origin.
      - Revolute Joint 2: rotation about Y, with a translation [0, 0, 0.39225] from Joint 1.
      - Revolute Joint 3: rotation about Z, with a translation [0, 0.093, 0] from Joint 2.
      - Revolute Joint 4: rotation about Y, with a translation [0, 0, 0.09465] from Joint 3.
      - TCP: translation [0, 0.0823, 0] and a constant orientation offset about Z of +1.570796325.
      
    The forward kinematics (position) can be derived as:
      p_x = d2*sin(θ1) + d4*sin(θ1+θ2) - d_tcp*sin(θ3)*cos(θ1+θ2)
      p_y = d3 + d_tcp*cos(θ3)
      p_z = d2*cos(θ1) + d4*cos(θ1+θ2) + d_tcp*sin(θ3)*sin(θ1+θ2)
    
    Note: The orientation of the TCP is given by:
      R_total = Ry(θ1+θ2) · Rz(θ3) · Ry(θ4) · Rz(psi)
    where psi = 1.570796325 is the fixed TCP yaw-offset.
    The target orientation is provided as roll, pitch, yaw in radians.
    (Here we assume the URDF convention: R_target = Rz(yaw) · Ry(pitch) · Rx(roll).)
    
    Due to inherent multiple solutions, this implementation generates all four candidate
    solutions (from the two choices in θ3 and the two choices in θ1) and then selects
    the candidate whose forward kinematics orientation best matches the target.
    
    :param p: Target TCP position [x, y, z].
    :param r: Target TCP orientation in rpy [roll, pitch, yaw] (radians).
    :return: Tuple (theta1, theta2, theta3, theta4) representing the joint angles in radians.
    """
    d2 = 0.39225
    d3 = 0.093
    d4 = 0.09465
    d_tcp = 0.0823
    psi = 1.570796325
    p = np.array(p)
    p_x, p_y, p_z = p

    def rot_x(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    cos_theta3 = (p_y - d3) / d_tcp
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3_pos = np.arccos(cos_theta3)
    theta3_neg = -theta3_pos
    r_target = np.sqrt(p_x ** 2 + p_z ** 2)
    δ = np.arctan2(p_x, p_z)
    candidates = []
    for theta3_candidate in [theta3_pos, theta3_neg]:
        sin_theta3 = np.sin(theta3_candidate)
        R_eff = np.sqrt(d4 ** 2 + (d_tcp * sin_theta3) ** 2)
        φ = np.arctan2(d_tcp * sin_theta3, d4)
        cos_term = (r_target ** 2 + d2 ** 2 - R_eff ** 2) / (2 * d2 * r_target)
        cos_term = np.clip(cos_term, -1.0, 1.0)
        theta1_offset = np.arccos(cos_term)
        for theta1_candidate in [δ + theta1_offset, δ - theta1_offset]:
            Vx = p_x - d2 * np.sin(theta1_candidate)
            Vz = p_z - d2 * np.cos(theta1_candidate)
            theta12 = np.arctan2(Vx, Vz) + φ
            theta2_candidate = theta12 - theta1_candidate
            R_pre = rot_y(theta1_candidate + theta2_candidate) @ rot_z(theta3_candidate)
            R_y_theta4 = R_pre.T @ R_target @ rot_z(-psi)
            theta4_candidate = np.arctan2(R_y_theta4[0, 2], R_y_theta4[0, 0])
            candidate = (theta1_candidate, theta2_candidate, theta3_candidate, theta4_candidate)
            R_forward = rot_y(theta1_candidate + theta2_candidate) @ rot_z(theta3_candidate) @ rot_y(theta4_candidate) @ rot_z(psi)
            err = np.linalg.norm(R_forward - R_target, ord='fro')
            candidates.append((err, candidate))
    best_candidate = min(candidates, key=lambda x: x[0])[1]
    return best_candidate