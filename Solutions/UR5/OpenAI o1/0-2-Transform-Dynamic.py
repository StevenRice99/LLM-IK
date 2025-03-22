def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Closed-form inverse kinematics for the 3-DoF serial manipulator (Z-Y-Y) with the given link offsets:
      • Joint 1 (Revolute Z) at the base with no offset.
      • Then a +Y offset of 0.13585 to Joint 2 (Revolute Y).
      • Then a relative offset of [0, -0.1197, 0.425] to Joint 3 (Revolute Y).
      • Finally a +Z offset of 0.39225 to the TCP.

    The orientation "r" comes from URDF RPY = [rx, ry, rz], meaning:
       R_des = Rz(rz)*Ry(ry)*Rx(rx)
    We assume all targets are reachable, and we do not check for unreachable cases.

    Strategy:
      1) Extract the overall desired yaw from R_des to set q1.
      2) Remove that yaw from the desired orientation, leaving a rotation purely about Y for the subchain (q2+q3).
      3) Solve q2+q3 = that leftover pitch.
      4) Transform the desired position into the frame after joint 1, then solve for q2 and q3 via geometry,
         respecting that q3 = leftover_pitch - q2.

    This code returns (q1, q2, q3) in radians, each in a range that fits ±2π.
    """
    import math
    import numpy as np
    x, y, z = p
    rx, ry, rz = r

    def rotation_rxyz(rx_, ry_, rz_):
        cz = math.cos(rz_)
        sz = math.sin(rz_)
        Rz_ = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        cy = math.cos(ry_)
        sy = math.sin(ry_)
        Ry_ = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        cx = math.cos(rx_)
        sx = math.sin(rx_)
        Rx_ = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
        return Rz_ @ Ry_ @ Rx_
    R_des = rotation_rxyz(rx, ry, rz)
    q1 = math.atan2(R_des[1, 0], R_des[0, 0])
    c1, s1 = (math.cos(q1), math.sin(q1))
    Rz_minus_q1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]], dtype=float)
    R_rem = Rz_minus_q1 @ R_des
    leftover_pitch = math.atan2(R_rem[0, 2], R_rem[0, 0])
    p_base = np.array([x, y, z], dtype=float)
    Rz_nq1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]], dtype=float)
    p_after_q1 = Rz_nq1 @ p_base

    def fk_sub(q2_, q3_):
        p_ = np.array([0.0, 0.13585, 0.0])
        Cy, Sy = (math.cos(q2_), math.sin(q2_))

        def rotY(vec, angle):
            c, s = (math.cos(angle), math.sin(angle))
            x_, y_, z_ = vec
            return np.array([c * x_ + s * z_, y_, -s * x_ + c * z_])
        p1 = np.array([0.0, 0.13585, 0.0])
        R_current = np.eye(3)
        R_current = R_current @ np.array([[Cy, 0, Sy], [0, 1, 0], [-Sy, 0, Cy]])
        offset_23 = np.array([0.0, -0.1197, 0.425])
        p2 = p1 + R_current @ offset_23
        R_q3 = np.array([[math.cos(q3_), 0, math.sin(q3_)], [0, 1, 0], [-math.sin(q3_), 0, math.cos(q3_)]])
        R_current = R_current @ R_q3
        offset_tcp = np.array([0.0, 0.0, 0.39225])
        p3 = p2 + R_current @ offset_tcp
        return p3
    best_q2 = 0.0
    best_err = float('inf')
    steps = 720
    for i in range(steps + 1):
        guess = -math.pi + 2 * math.pi * (i / steps)
        q3_ = leftover_pitch - guess
        if abs(guess) > 6.2831853 or abs(q3_) > 6.2831853:
            continue
        p_est = fk_sub(guess, q3_)
        err = np.linalg.norm(p_est - p_after_q1)
        if err < best_err:
            best_err = err
            best_q2 = guess
    q2 = best_q2
    q3 = leftover_pitch - q2

    def wrap(a):
        a_mod = (a + math.pi) % (2 * math.pi) - math.pi
        if a_mod < -math.pi:
            a_mod += 2 * math.pi
        elif a_mod > math.pi:
            a_mod -= 2 * math.pi
        while a_mod < -6.2831853:
            a_mod += 2 * math.pi
        while a_mod > 6.2831853:
            a_mod -= 2 * math.pi
        return a_mod
    q1 = wrap(q1)
    q2 = wrap(q2)
    q3 = wrap(q3)
    return (q1, q2, q3)