import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the closed‐form inverse kinematics solution for a 3DOF serial manipulator.
    The robot has:
      – Joint 1: Revolute about Z at the base.
      – Joint 2: Revolute about Y; its origin is offset by [0, 0.13585, 0] in joint1 frame.
      – Joint 3: Revolute about Y; its origin is offset by [0, -0.1197, 0.425] in joint2 frame;
                 only the 0.425 (along the direction orthogonal to the joint’s axis) is used.
      – TCP: Offset [0, 0, 0.39225] from joint3.
      
    The provided TCP target is (p) and its orientation in rpy is (r).
    The decoupling is performed by:
      1. Computing the wrist center (WC) by subtracting the TCP offset (in the TCP’s z-axis)
         from the given TCP position.
      2. Determining joint1 as the angle of the WC projection on the xy–plane.
      3. Expressing the WC in the frame of joint2 (subtracting off the joint2 offset rotated by q1).
      4. Extracting the effective rotation (theta_total = q2+q3) from the TCP orientation.
         Here the TCP rotation matrix is computed from the rpy angles using 
         R_tcp = Rz(rz)*Ry(ry)*Rx(rx).
         Then we remove the base rotation with Rz(–q1) so that the effective rotation 
         corresponds to the 2R planar chain.
      5. With effective link lengths L1 = 0.425 and L2 = 0.39225,
         we solve the 2R IK by “shifting” the wrist target by the TCP offset effect.
         That is, in joint2’s (and base’s) x–z plane the wrist target (P) is:
           P = WC – (p_joint2),  where p_joint2 = Rz(q1)*[0, d12, 0] with d12 = 0.13585.
         Then, letting
           A = P_x – L2*sin(theta_total)
           B = P_z – L2*cos(theta_total)
         we set q2 = atan2(A, B) and q3 = theta_total – q2.
    Note:
      – Throughout, it is assumed that the given target is reachable.
      – All angles are in radians.
      
    :param p: Desired TCP position [x, y, z] in base coordinates.
    :param r: Desired TCP orientation (roll, pitch, yaw) in radians.
    :return: (q1, q2, q3) joint angles in radians.
    """

    def rot_x(a: float):
        ca = math.cos(a)
        sa = math.sin(a)
        return [[1, 0, 0], [0, ca, -sa], [0, sa, ca]]

    def rot_y(a: float):
        ca = math.cos(a)
        sa = math.sin(a)
        return [[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]

    def rot_z(a: float):
        ca = math.cos(a)
        sa = math.sin(a)
        return [[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]]

    def mat_mult(A, B):
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                result[i][j] = sum((A[i][k] * B[k][j] for k in range(3)))
        return result

    def mat_vec_mult(A, v):
        return [A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2], A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2], A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]]

    def transpose(M):
        return [[M[j][i] for j in range(3)] for i in range(3)]
    x, y, z = p
    rx, ry, rz = r
    Rz_mat = rot_z(rz)
    Ry_mat = rot_y(ry)
    Rx_mat = rot_x(rx)
    R_temp = mat_mult(Ry_mat, Rx_mat)
    R_tcp = mat_mult(Rz_mat, R_temp)
    d_tcp = 0.39225
    n_tcp = [R_tcp[0][2], R_tcp[1][2], R_tcp[2][2]]
    WC = (x - d_tcp * n_tcp[0], y - d_tcp * n_tcp[1], z - d_tcp * n_tcp[2])
    q1 = math.atan2(WC[1], WC[0])
    d12 = 0.13585
    p_joint2 = (-math.sin(q1) * d12, math.cos(q1) * d12, 0.0)
    P = (WC[0] - p_joint2[0], WC[1] - p_joint2[1], WC[2] - p_joint2[2])
    P_x = P[0]
    P_z = P[2]
    Rz_minus_q1 = rot_z(-q1)
    R_wrist = mat_mult(Rz_minus_q1, R_tcp)
    theta_total = math.atan2(R_wrist[0][2], R_wrist[2][2])
    L1 = 0.425
    L2 = 0.39225
    A = P_x - L2 * math.sin(theta_total)
    B = P_z - L2 * math.cos(theta_total)
    q2 = math.atan2(A, B)
    q3 = theta_total - q2
    return (q1, q2, q3)