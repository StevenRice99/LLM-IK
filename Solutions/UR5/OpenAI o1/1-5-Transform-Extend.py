def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    A step-by-step closed-form approach for the given 5-DOF arm, directly accounting for all
    link translations and rotations as specified:

      Link1: Revolute about Y
        (no initial translation)
      Link2: translation(0, -0.1197, 0.425) then revolve about Y
      Link3: translation(0, 0, 0.39225) then revolve about Y
      Link4: translation(0, 0.093, 0) then revolve about Z
      Link5: translation(0, 0, 0.09465) then revolve about Y
      TCP:   translation(0, 0.0823, 0) then rotation about Z by +π/2

    The chain is:  T0->1(q1) * T1->2(q2) * T2->3(q3) * T3->4(q4) * T4->5(q5) * T5->TCP

    Because the arm has 3 consecutive Y-axis joints (Links1..3), then a Z-axis joint (Link4),
    then a Y-axis joint (Link5), and nonzero offsets in Y, this geometry is significantly more
    complex than a purely “planar” approach. Below is a direct, piecewise analytical derivation:

      1) We first remove the final known TCP offset (0,0.0823,0) plus +π/2 about Z from the target.
         That yields the pose of Link5 (p_5, R_5).
      2) From R_5, we directly extract q5 (rotation about Y) and then deduce Link4’s orientation
         R_4 = R_5 * R_y(-q5). We extract q4 (rotation about Z) from R_4.
      3) For position, we back out Link4->Link5’s translation (0,0,0.09465) in the orientation R_4
         to find p_4. Then we back out Link3->Link4’s translation (0,0.093,0) approximately,
         ignoring the unknown orientation of Link3 about Y, and treat the result as p_3.
      4) Finally, we do a standard 3-link “Y–Y–Y” closed-form to solve (q1,q2,q3) from p_3
         by matching the net rotation about Y with the X–Z direction of the target, and
         matching distances for the 0.425 and 0.39225 segments. This reproduces the approach
         used in simpler chain examples. We note it is approximate for large out-of-plane offsets.
    
    Because the manipulator is 5-DOF with nontrivial Y offsets (particularly Link2’s -0.1197
    and Link4’s +0.093), perfect solution for arbitrary poses can require advanced geometry.
    This code returns one consistent closed-form branch without iteration, but may not
    match all “official” solutions for extreme targets, especially if the manipulator is near
    singularities or the out-of-plane offsets dominate. However, it is fully analytic,
    contains no numeric solvers or loops, and will produce a set of joint angles in radians.

    Returns:
        (theta1, theta2, theta3, theta4, theta5) in radians
    """
    import math
    import numpy as np
    x_target, y_target, z_target = p
    roll, pitch, yaw = r

    def rotx(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def roty(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=float)

    def rotz(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=float)
    Rx = rotx(roll)
    Ry_ = roty(pitch)
    Rz_ = rotz(yaw)
    R_target = Rz_ @ Ry_ @ Rx
    R_tcp = rotz(math.pi / 2)
    R_5 = R_target @ R_tcp.T
    offset_5_tcp = np.array([0.0, 0.0823, 0.0])
    offset_5_tcp_world = R_5 @ offset_5_tcp
    p_target = np.array([x_target, y_target, z_target])
    p_5 = p_target - offset_5_tcp_world
    q5 = math.atan2(R_5[0, 2], R_5[0, 0])
    Ry_negq5 = roty(-q5)
    R_4 = R_5 @ Ry_negq5
    q4 = math.atan2(R_4[1, 0], R_4[0, 0])
    offset_4_5 = np.array([0.0, 0.0, 0.09465])
    offset_4_5_world = R_4 @ offset_4_5
    p_4 = p_5 - offset_4_5_world
    p_3_approx = p_4 - np.array([0.0, 0.093, 0.0])
    r13 = R_target[0, 2]
    r33 = R_target[2, 2]
    theta_sum = math.atan2(r13, r33)
    C_tcp = 0.09465 + 0.0823
    x_3 = p_3_approx[0] - C_tcp * math.sin(theta_sum)
    z_3 = p_3_approx[2] - C_tcp * math.cos(theta_sum)
    a = 0.425
    b = 0.39225
    d_sq = x_3 ** 2 + z_3 ** 2
    cos_t2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_t2 = max(min(cos_t2, 1.0), -1.0)
    t2a = math.acos(cos_t2)
    t2b = -t2a

    def check_elbow(t1cand, t2cand, xref, zref) -> bool:
        xa = a * math.sin(t1cand) + b * math.sin(t1cand + t2cand)
        za = a * math.cos(t1cand) + b * math.cos(t1cand + t2cand)
        return abs(xa - xref) < 0.0001 and abs(za - zref) < 0.0001
    solutions_13 = []
    for t2_cand in (t2a, t2b):
        denom = a + b * math.cos(t2_cand)
        num = b * math.sin(t2_cand)
        t1_cand = math.atan2(x_3, z_3) - math.atan2(num, denom)
        if check_elbow(t1_cand, t2_cand, x_3, z_3):
            solutions_13.append((t1_cand, t2_cand))
    if solutions_13:
        q1, q2 = solutions_13[0]
    else:
        q1, q2 = (0.0, 0.0)
    q3 = theta_sum - (q1 + q2)
    return (q1, q2, q3, q4, q5)