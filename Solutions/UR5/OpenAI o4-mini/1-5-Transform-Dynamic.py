import numpy as np
import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Analytical IK for 5‑DOF Y–Y–Y–Z–Y arm:
      Joint1: Y at [0,0,0]
      Joint2: Y at [0,-0.1197,0.425]
      Joint3: Y at [0,0,0.39225]
      Joint4: Z at [0,0.093,0]
      Joint5: Y at [0,0,0.09465]
      TCP   at [0,0.0823,0] + yaw offset +90° about Z
    :param p:  target TCP position (x,y,z)
    :param r:  target TCP rpy (roll,pitch,yaw)
    :return: (q1,q2,q3,q4,q5) in radians
    """
    link2_off = np.array([0.0, -0.1197, 0.425])
    d2 = 0.39225
    d3 = 0.093
    d4 = 0.09465
    d_tcp = 0.0823
    psi = 1.570796325

    def rot_x(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    roll, pitch, yaw = r
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    P = np.array(p)
    wrist_world = P - d_tcp * R_target[:, 1]
    wx, wy, wz = wrist_world
    q1 = math.atan2(wx, wz)
    R1_inv = rot_y(-q1)
    wrist_1 = R1_inv @ wrist_world
    wrist_2 = wrist_1 - link2_off
    x2, y2, z2 = wrist_2
    a1 = d2
    a2 = d4
    r_planar = math.hypot(x2, z2)
    cos_q3 = (r_planar ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_candidates = [math.acos(cos_q3), -math.acos(cos_q3)]
    best_err = 1000000000.0
    best_sol = (0.0, 0.0, 0.0, 0.0, 0.0)
    R2_target = R1_inv @ R_target
    for q3 in q3_candidates:
        gamma = math.atan2(a2 * math.sin(q3), a1 + a2 * math.cos(q3))
        q2 = math.atan2(x2, z2) - gamma
        R2_5 = R2_target @ rot_z(-psi)
        R_pre = rot_y(q2) @ rot_y(q3)
        R34_45 = R_pre.T @ R2_5
        m00 = R34_45[0, 0]
        m10 = R34_45[1, 0]
        m02 = R34_45[0, 2]
        q4 = math.atan2(m10, m00)
        q5 = math.atan2(m02, m00)
        Rf = rot_y(q1) @ rot_y(q2) @ rot_y(q3) @ rot_z(q4) @ rot_y(q5) @ rot_z(psi)
        err = np.linalg.norm(Rf - R_target, ord='fro')
        if err < best_err:
            best_err = err
            best_sol = (q1, q2, q3, q4, q5)
    return best_sol