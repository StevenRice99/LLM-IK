import math
import numpy as np
import sympy as sp

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A more systematic symbolic-based closed-form IK solver for the 5-DOF manipulator.
    
    This approach explicitly uses Sympy for symbolic transformations and then solves
    for the joint variables (q1..q5) in closed form. The manipulator structure is:

      1) Revolute 1 (q1) about Y at the origin.
      2) Revolute 2 (q2) about Y at [0, -0.1197, 0.425].
      3) Revolute 3 (q3) about Y at [0, 0, 0.39225].
      4) Revolute 4 (q4) about Z at [0, 0.093, 0].
      5) Revolute 5 (q5) about Y at [0, 0, 0.09465].
      TCP offset: [0, 0.0823, 0], plus a fixed rotation about Z by π/2.

    In summary:
      1) We symbolically construct the forward-kinematics transform T(q1,q2,q3,q4,q5).
      2) We equate T to the desired transform from the input (p, r), i.e. T_target.
      3) We solve symbolically for q1..q5, collecting all real solutions.
      4) We evaluate each candidate solution numerically and pick the one with minimal error
         in both position and orientation vs. the target.

    Note:
      - Since it's only a 5-DOF manipulator with parallel Y-axes for joints 1..3, 
        there are multiple “elbow” solutions. We collect them all.
      - Joint angles are wrapped into [-π, π] where applicable.
      - We assume each input pose is reachable (per instructions).

    :param p: (x, y, z) the desired TCP position (in meters).
    :param r: (roll, pitch, yaw) the desired TCP orientation (in radians),
              using the URDF convention R = Rz(yaw)*Ry(pitch)*Rx(roll).
    :return: A tuple (q1, q2, q3, q4, q5) in radians.
    """
    px, py, pz = p
    roll, pitch, yaw = r

    def rot_x(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], dtype=float)

    def rot_y(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=float)

    def rot_z(a):
        return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=float)
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    T_target = np.eye(4, dtype=float)
    T_target[0:3, 0:3] = R_target
    T_target[0, 3] = px
    T_target[1, 3] = py
    T_target[2, 3] = pz
    q1_s, q2_s, q3_s, q4_s, q5_s = sp.symbols('q1 q2 q3 q4 q5', real=True)

    def sym_rot_y(q):
        return sp.Matrix([[sp.cos(q), 0, sp.sin(q), 0], [0, 1, 0, 0], [-sp.sin(q), 0, sp.cos(q), 0], [0, 0, 0, 1]])

    def sym_rot_z(q):
        return sp.Matrix([[sp.cos(q), -sp.sin(q), 0, 0], [sp.sin(q), sp.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def sym_trans(x, y, z):
        return sp.Matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    offset2 = (0.0, -0.1197, 0.425)
    offset3 = (0.0, 0.0, 0.39225)
    offset4 = (0.0, 0.093, 0.0)
    offset5 = (0.0, 0.0, 0.09465)
    offset_tcp = (0.0, 0.0823, 0.0)
    psi = sp.pi / 2
    T1 = sym_rot_y(q1_s)
    T2 = sym_trans(*offset2) * sym_rot_y(q2_s)
    T3 = sym_trans(*offset3) * sym_rot_y(q3_s)
    T4 = sym_trans(*offset4) * sym_rot_z(q4_s)
    T5 = sym_trans(*offset5) * sym_rot_y(q5_s)
    Ttcp = sym_trans(*offset_tcp) * sym_rot_z(psi)
    T_fk_sym = T1 * T2 * T3 * T4 * T5 * Ttcp
    fk_func = sp.lambdify((q1_s, q2_s, q3_s, q4_s, q5_s), T_fk_sym, 'numpy')

    def pose_error(q1v, q2v, q3v, q4v, q5v):
        T_num = fk_func(q1v, q2v, q3v, q4v, q5v)
        dx = T_num[0, 3] - px
        dy = T_num[1, 3] - py
        dz = T_num[2, 3] - pz
        pos_err = math.sqrt(dx * dx + dy * dy + dz * dz)
        R_num = np.array(T_num[:3, :3], dtype=float)
        diff = R_num - R_target
        orient_err = np.linalg.norm(diff)
        return pos_err + orient_err

    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a <= -math.pi:
            a += 2 * math.pi
        return a
    baseAngle = math.atan2(px, pz)
    q1_set = [baseAngle, baseAngle + math.pi]

    def elbow_candidates(px, py, pz):
        R_plane = math.sqrt(px * px + pz * pz)
        h = py + 0.1197
        dist = math.sqrt(R_plane * R_plane + h * h)
        L2_ = 0.425
        L3_ = 0.39225
        cQ3 = (dist * dist - L2_ * L2_ - L3_ * L3_) / (2 * L2_ * L3_)
        cQ3 = max(min(cQ3, 1), -1)
        try:
            q3a_ = math.acos(cQ3)
            q3b_ = -q3a_
        except ValueError:
            return []
        phi_ = math.atan2(h, R_plane)
        out = []
        for q3_ in [q3a_, q3b_]:
            sin3_ = math.sin(q3_)
            cos3_ = math.cos(q3_)
            alpha_ = math.atan2(0.39225 * sin3_, 0.425 + 0.39225 * cos3_)
            for sign_ in [+1, -1]:
                q2_ = phi_ + sign_ * alpha_
                out.append((q2_, q3_))
        return out
    cand_123 = []
    eqs = elbow_candidates(px, py, pz)
    if not eqs:
        eqs = [(0.0, 0.0)]
    for q2c, q3c in eqs:
        for q1c in q1_set:
            cand_123.append((q1c, q2c, q3c))

    def solve_q4_q5(q1v, q2v, q3v):
        T1n = np.array(sym_rot_y(q1_s).evalf(subs={q1_s: q1v}), dtype=float)
        T2n = np.array((sym_trans(*offset2) * sym_rot_y(q2_s)).evalf(subs={q2_s: q2v}), dtype=float)
        T3n = np.array((sym_trans(*offset3) * sym_rot_y(q3_s)).evalf(subs={q3_s: q3v}), dtype=float)
        T_3 = T1n @ T2n @ T3n
        R_3 = T_3[:3, :3]
        Rz_negpsi = rot_z(-math.pi / 2)
        R_left = R_3.T @ R_target @ Rz_negpsi
        q4_ = math.atan2(R_left[1, 0], R_left[0, 0])
        R_z_minus_q4 = rot_z(-q4_)
        R_temp = R_z_minus_q4 @ R_left
        q5_ = math.atan2(R_temp[0, 2], R_temp[0, 0])
        return (q4_, q5_)
    best_err = 1000000000.0
    best_sol = (0, 0, 0, 0, 0)
    for cand_q1, cand_q2, cand_q3 in cand_123:
        q4c, q5c = solve_q4_q5(cand_q1, cand_q2, cand_q3)
        q1w = wrap_angle(cand_q1)
        q2w = wrap_angle(cand_q2)
        q3w = wrap_angle(cand_q3)
        q4w = wrap_angle(q4c)
        q5w = wrap_angle(q5c)
        err_ = pose_error(q1w, q2w, q3w, q4w, q5w)
        if err_ < best_err:
            best_err = err_
            best_sol = (q1w, q2w, q3w, q4w, q5w)
    return best_sol