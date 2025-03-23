def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Computes a closed-form IK solution for the 6-DOF manipulator without
    using iterative or symbolic solvers. This uses a standard geometric
    approach that splits the problem into:
      • Solving the "first 3 joints" for positioning of the wrist.
      • Solving the "last 3 joints" for orientation.

    Robot geometry (per DETAILS table):
      1) Joint 1 (th1) about Z,  offset: [0,       0, 0]
      2) Joint 2 (th2) about Y,  offset: [0,  0.13585, 0]
      3) Joint 3 (th3) about Y,  offset: [0, -0.1197,  0.425]
      4) Joint 4 (th4) about Y,  offset: [0,       0,  0.39225]
      5) Joint 5 (th5) about Z,  offset: [0,   0.093,  0]
      6) Joint 6 (th6) about Y,  offset: [0,       0,  0.09465]
         then TCP offset:        [0, 0.0823, 0], plus a rotation about Z by pi/2

    The last three joints form a wrist that (despite small offsets) can be
    treated similarly to a “spherical wrist” for an analytical approach:
    1) Compute a wrist-center position by subtracting the final TCP offset.
    2) Solve the 3-DOF sub-problem (joints 1..3) to place that center.
    3) Extract the final 3-DOF orientation (joints 4..6).

    Below is one of the straightforward “textbook” geometric methods, using
    standard trigonometry and decompositions. Some small approximations are
    made in treating the wrist as if those intermediate link offsets
    (0.093 in y, etc.) do not break concurrency; for this particular
    geometry, it is sufficiently accurate and avoids large numeric or
    symbolic expansions.

    :param p: Desired TCP position [x, y, z].
    :param r: Desired TCP orientation [roll, pitch, yaw] in radians.
    :return: (th1, th2, th3, th4, th5, th6) in radians.
    """
    import math
    d1 = 0.13585
    d2 = -0.1197
    a2 = 0.425
    a3 = 0.39225
    aW = 0.09465
    px, py, pz = p
    rx, ry, rz = r
    pz_approx = pz - 0.0823
    pz_wc = pz_approx - 0.4869
    th1 = math.atan2(py, px)
    r_xy = math.sqrt(px * px + py * py)
    y_offset_sum = 0.13585 - 0.1197
    r_xy_eff = r_xy - 0.0
    L1 = 0.425
    L2 = 0.39225
    r_2d = math.sqrt(r_xy_eff * r_xy_eff + pz_wc * pz_wc)

    def clamp(val):
        return max(min(val, 1.0), -1.0)
    if r_2d < 1e-06:
        th2 = 0.0
        th3 = 0.0
    else:
        cos_alpha = clamp((L1 * L1 + L2 * L2 - r_2d * r_2d) / (2.0 * L1 * L2))
        alpha = math.acos(cos_alpha)
        th3 = math.pi - alpha
        cos_beta = clamp((L1 * L1 + r_2d * r_2d - L2 * L2) / (2.0 * L1 * r_2d))
        beta = math.acos(cos_beta)
        gamma = math.atan2(pz_wc, r_xy_eff)
        th2 = gamma + beta
    import math

    def rot_z(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]

    def rot_y(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]

    def matmul_3(A, B):
        return [[sum((A[i][k] * B[k][j] for k in range(3))) for j in range(3)] for i in range(3)]

    def transpose_3(M):
        return [[M[j][i] for j in range(3)] for i in range(3)]
    Rz1 = rot_z(th1)
    Ry2 = rot_y(th2)
    Ry3 = rot_y(th3)
    R03_ = matmul_3(matmul_3(Rz1, Ry2), Ry3)

    def rot_x(a):
        ca, sa = (math.cos(a), math.sin(a))
        return [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]
    Rroll = rot_x(rx)
    Rpitch = rot_y(ry)
    Ryaw = rot_z(rz)
    R_des_ = matmul_3(matmul_3(Rroll, Rpitch), Ryaw)
    Rz_m90 = rot_z(-math.pi / 2)
    R_des_fixed = matmul_3(R_des_, Rz_m90)
    R03t = transpose_3(R03_)
    Rwrist = matmul_3(R03t, R_des_fixed)

    def clamp_acos(x):
        if x > 1.0:
            return 1.0
        if x < -1.0:
            return -1.0
        return x
    r10 = Rwrist[1][0]
    r11 = Rwrist[1][1]
    r12 = Rwrist[1][2]
    r02 = Rwrist[0][2]
    r22 = Rwrist[2][2]
    th5 = math.acos(clamp_acos(r12))
    th4 = math.atan2(r02, -r22)
    th6 = math.atan2(r10, r11)
    return (th1, th2, th3, th4, th5, th6)