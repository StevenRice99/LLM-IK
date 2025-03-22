import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Attempts a closed-form solution for a 3-DOF serial chain:
       • Joint 1 (about Z)
       • Joint 2 (about Y)
       • Joint 3 (about Y)
    using the link geometry from the "DETAILS":
       • Joint1 at [0,0,0], revolve Z
       • Joint2 at [0,0.13585,0], revolve Y
       • Joint3 at [0,-0.1197,0.425], revolve Y
       • TCP   at [0,0,0.39225]
    The orientation r is given as RPY [rx, ry, rz].
    Because the robot has only yaw (q1) and two pitches (q2,q3),
    we interpret:
       q1 ≈ final yaw = rz
       q2 + q3 ≈ final pitch = ry
    Possibly plus integer multiples of π due to the many valid IK branches.
    
    This solver:
      1) Considers two main candidates for q1 = rz or (rz ± π) to handle
         alternative orientations about Z that might better match the target.
      2) For each candidate q1, compute α = ry. Then the local geometry for
         joints 2 and 3 remains the same: q2 + q3 = α.
      3) Rotate the target position p by Rz(-q1) to find the local position,
         subtract the link-2 offset [0,0.13585,0].
      4) Solve the planar system for q2 from the partial geometry:
           x_local = 0.425 sin(q2) + 0.39225 sin(α)
           z_local = 0.425 cos(q2) + 0.39225 cos(α)
         => q2 = atan2( x_local - 0.39225 sin(α),
                        z_local - 0.39225 cos(α) )
         => q3 = α - q2
      5) Do a simple forward check for each candidate and pick whichever end
         position is closer to the target. (Orientation is automatically matched
         about Y,Z in this model if there's a feasible solution.)
    
    Returns (q1,q2,q3). All angles in radians, without any final wrapping or limit checks.
    """

    def forward_kin(q1, q2, q3):
        """Compute the end-effector position (x,y,z) using the given angles."""
        c1, s1 = (math.cos(q1), math.sin(q1))
        px1 = 0.0
        py1 = 0.13585
        pz1 = 0.0

        def rotZ(th):
            c, s = (math.cos(th), math.sin(th))
            return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

        def rotY(th):
            c, s = (math.cos(th), math.sin(th))
            return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

        def mat_vec(R, v):
            return [R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2], R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2], R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2]]
        Rw = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        pw = [0, 0, 0]
        R1 = rotZ(q1)
        pw = [pw[0], pw[1], pw[2]]
        Rw = [[Rw[0][0] * R1[0][0] + Rw[0][1] * R1[1][0] + Rw[0][2] * R1[2][0], Rw[0][0] * R1[0][1] + Rw[0][1] * R1[1][1] + Rw[0][2] * R1[2][1], Rw[0][0] * R1[0][2] + Rw[0][1] * R1[1][2] + Rw[0][2] * R1[2][2]], [Rw[1][0] * R1[0][0] + Rw[1][1] * R1[1][0] + Rw[1][2] * R1[2][0], Rw[1][0] * R1[0][1] + Rw[1][1] * R1[1][1] + Rw[1][2] * R1[2][1], Rw[1][0] * R1[0][2] + Rw[1][1] * R1[1][2] + Rw[1][2] * R1[2][2]], [Rw[2][0] * R1[0][0] + Rw[2][1] * R1[1][0] + Rw[2][2] * R1[2][0], Rw[2][0] * R1[0][1] + Rw[2][1] * R1[1][1] + Rw[2][2] * R1[2][1], Rw[2][0] * R1[0][2] + Rw[2][1] * R1[1][2] + Rw[2][2] * R1[2][2]]]
        off2 = [0, 0.13585, 0]
        off2_w = mat_vec(Rw, off2)
        pw = [pw[0] + off2_w[0], pw[1] + off2_w[1], pw[2] + off2_w[2]]
        R2 = rotY(q2)
        Rw2 = [[Rw[0][0] * R2[0][0] + Rw[0][1] * R2[1][0] + Rw[0][2] * R2[2][0], Rw[0][0] * R2[0][1] + Rw[0][1] * R2[1][1] + Rw[0][2] * R2[2][1], Rw[0][0] * R2[0][2] + Rw[0][1] * R2[1][2] + Rw[0][2] * R2[2][2]], [Rw[1][0] * R2[0][0] + Rw[1][1] * R2[1][0] + Rw[1][2] * R2[2][0], Rw[1][0] * R2[0][1] + Rw[1][1] * R2[1][1] + Rw[1][2] * R2[2][1], Rw[1][0] * R2[0][2] + Rw[1][1] * R2[1][2] + Rw[1][2] * R2[2][2]], [Rw[2][0] * R2[0][0] + Rw[2][1] * R2[1][0] + Rw[2][2] * R2[2][0], Rw[2][0] * R2[0][1] + Rw[2][1] * R2[1][1] + Rw[2][2] * R2[2][1], Rw[2][0] * R2[0][2] + Rw[2][1] * R2[1][2] + Rw[2][2] * R2[2][2]]]
        off3 = [0, -0.1197, 0.425]
        off3_w = mat_vec(Rw2, off3)
        pw2 = [pw[0] + off3_w[0], pw[1] + off3_w[1], pw[2] + off3_w[2]]
        R3 = rotY(q3)
        Rw3 = [[Rw2[0][0] * R3[0][0] + Rw2[0][1] * R3[1][0] + Rw2[0][2] * R3[2][0], Rw2[0][0] * R3[0][1] + Rw2[0][1] * R3[1][1] + Rw2[0][2] * R3[2][1], Rw2[0][0] * R3[0][2] + Rw2[0][1] * R3[1][2] + Rw2[0][2] * R3[2][2]], [Rw2[1][0] * R3[0][0] + Rw2[1][1] * R3[1][0] + Rw2[1][2] * R3[2][0], Rw2[1][0] * R3[0][1] + Rw2[1][1] * R3[1][1] + Rw2[1][2] * R3[2][1], Rw2[1][0] * R3[0][2] + Rw2[1][1] * R3[1][2] + Rw2[1][2] * R3[2][2]], [Rw2[2][0] * R3[0][0] + Rw2[2][1] * R3[1][0] + Rw2[2][2] * R3[2][0], Rw2[2][0] * R3[0][1] + Rw2[2][1] * R3[1][1] + Rw2[2][2] * R3[2][1], Rw2[2][0] * R3[0][2] + Rw2[2][1] * R3[1][2] + Rw2[2][2] * R3[2][2]]]
        off_tcp = [0, 0, 0.39225]
        off_tcp_w = mat_vec(Rw3, off_tcp)
        pw3 = [pw2[0] + off_tcp_w[0], pw2[1] + off_tcp_w[1], pw2[2] + off_tcp_w[2]]
        return pw3
    xT, yT, zT = p
    rx, ry, rz = r
    alpha = ry
    candidates_q1 = [rz]
    candidates_q1.append(rz + math.pi)
    best_sol = None
    best_err = 1000000000.0
    for test_q1 in candidates_q1:
        c1 = math.cos(test_q1)
        s1 = math.sin(test_q1)
        x_local = xT * c1 + yT * s1
        y_local = -xT * s1 + yT * c1
        z_local = zT
        y_local -= 0.13585
        Xp = x_local - 0.39225 * math.sin(alpha)
        Zp = z_local - 0.39225 * math.cos(alpha)
        test_q2 = math.atan2(Xp, Zp)
        test_q3 = alpha - test_q2
        px, py, pz = forward_kin(test_q1, test_q2, test_q3)
        err = math.sqrt((px - xT) ** 2 + (py - yT) ** 2 + (pz - zT) ** 2)
        if err < best_err:
            best_err = err
            best_sol = (test_q1, test_q2, test_q3)
    if best_sol is None:
        return (rz, 0.0, ry)
    else:
        return best_sol