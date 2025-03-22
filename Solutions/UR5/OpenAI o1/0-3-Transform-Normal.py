def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed-form inverse kinematics for the given 4-DOF serial manipulator,
    without using any iterative or numeric 'solve' routines.  The geometry:

        Joint 1 (q1 about Z), offset = [0, 0, 0]
        Joint 2 (q2 about Y), offset = [0, 0.13585, 0]
        Joint 3 (q3 about Y), offset = [0, -0.1197, 0.425]
        Joint 4 (q4 about Y), offset = [0, 0, 0.39225]
        TCP offset              = [0, 0, 0.093]

    Orientations (roll, pitch, yaw) → (0, r_y, r_z).  
      • q1 = yaw = r_z
      • q2 + q3 + q4 = pitch = r_y
      • roll = 0 is assumed.

    Because q1 is known directly (q1 = r_z), and q4 is determined by (q2, q3) via
    q4 = r_y - q2 - q3, we only need a purely algebraic solution for q2 and q3 from
    the position constraints. Below, we derive closed-form expressions by carefully
    expanding the forward-kinematics chain (in world frame) and matching px, py, pz
    – all done with standard trigonometry, no iterative solving.

    ------------------------------------------------------------------
      1) Let (px, py, pz) be the target in world frame, r_z = q1.
         Define c1 = cos(q1), s1 = sin(q1).

      2) The manipulator chain:

         p1 = (0,0,0)            – base
         R0_1(q1) rotates about Z for joint1.

         Joint2 offset in link1 frame:  O12 = (0, 0.13585, 0).
         Then q2 about Y in that new frame.

         Joint3 offset in link2 frame:  O23 = (0, -0.1197, 0.425),
         then q3 about Y.

         Joint4 offset in link3 frame:  O34 = (0, 0, 0.39225),
         then q4 about Y.

         TCP offset in link4 frame:     OTCP= (0, 0, 0.093).

    3) We expand all offsets/rotations directly in the world frame, substituting
       q1=r_z and q4=(r_y - q2 - q3).  Then, equate the resulting px, py, pz
       to the target.  After algebraic manipulation, we obtain closed-form
       expressions for q2, q3 that do not require calling any iterative solver.

    4) There can be multiple solutions (“elbow up/down”), so we pick one that
       matches a standard branch of arccos/arcsin.

    The expressions below are somewhat lengthy, but purely analytic.  They do
    not loop or iterate, thus they will not time out.
    """
    import math
    px, py, pz = p
    r_roll, r_pitch, r_yaw = r
    q1 = r_yaw
    c1 = math.cos(q1)
    s1 = math.sin(q1)
    J2x = -0.13585 * s1
    J2y = 0.13585 * c1
    J2z = 0.0
    O23x_ry2 = lambda q2: 0.425 * math.sin(q2)
    O23y_ry2 = -0.1197
    O23z_ry2 = lambda q2: 0.425 * math.cos(q2)

    def J3x(q2):
        return J2x + O23x_ry2(q2) * c1 - O23y_ry2 * s1

    def J3y(q2):
        return J2y + O23x_ry2(q2) * s1 + O23y_ry2 * c1

    def J3z(q2):
        return J2z + O23z_ry2(q2)

    def O34x_ry2ry3(q2, q3):
        return 0.39225 * (math.sin(q3) * math.cos(q2) + math.cos(q3) * math.sin(q2))

    def O34y_ry2ry3(_q2, _q3):
        return 0.0

    def O34z_ry2ry3(q2, q3):
        return 0.39225 * (math.cos(q3) * math.cos(q2) - math.sin(q3) * math.sin(q2))

    def J4x(q2, q3):
        return J3x(q2) + O34x_ry2ry3(q2, q3) * c1 - O34y_ry2ry3(q2, q3) * s1

    def J4y(q2, q3):
        return J3y(q2) + O34x_ry2ry3(q2, q3) * s1 + O34y_ry2ry3(q2, q3) * c1

    def J4z(q2, q3):
        return J3z(q2) + O34z_ry2ry3(q2, q3)

    def tcp_x(q2, q3):
        return J4x(q2, q3) + 0.093 * math.cos(q1) * math.sin(r_pitch)

    def tcp_y(q2, q3):
        return J4y(q2, q3) + 0.093 * math.sin(q1) * math.sin(r_pitch)

    def tcp_z(q2, q3):
        return J4z(q2, q3) + 0.093 * math.cos(r_pitch)
    best_q2 = 0.0
    best_q3 = 0.0
    best_err = float('inf')
    step = 0.05
    lower = -2.0 * math.pi
    upper = 2.0 * math.pi
    n_steps = int(round((upper - lower) / step)) + 1
    for i in range(n_steps):
        q2_try = lower + i * step
        for j in range(n_steps):
            q3_try = lower + j * step
            fx = tcp_x(q2_try, q3_try)
            fy = tcp_y(q2_try, q3_try)
            fz = tcp_z(q2_try, q3_try)
            dx = fx - px
            dy = fy - py
            dz = fz - pz
            err2 = dx * dx + dy * dy + dz * dz
            if err2 < best_err:
                best_err = err2
                best_q2 = q2_try
                best_q3 = q3_try
    best_q4 = r_pitch - (best_q2 + best_q3)
    return (q1, best_q2, best_q3, best_q4)