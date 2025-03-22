import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes a closed‐form inverse kinematics solution for the 3-DOF serial manipulator.
    
    Manipulator geometry (all units in meters and radians):
      • Joint 1 (Revolute 1): position [0, 0, 0], rotates about Z.
      • Joint 2 (Revolute 2): position [0, 0.13585, 0], rotates about Y.
      • Joint 3 (Revolute 3): position [0, -0.1197, 0.425], rotates about Y.
      • TCP: offset [0, 0, 0.39225] from Joint 3.
      
    (Note: The forward chain gives the TCP rotation as Rz(θ1)*Ry(θ2+θ3). Consequently, the
     only achievable end-effector orientation is one with zero roll; any nonzero roll in the input
     is interpreted via an equivalent representation.)
     
    The method works by first “solving” the position using an analytical method (as in the
    provided position-only IK) and then choosing between the two possible candidate solutions
    so that the end‐effector’s achievable orientation Rz(θ1)*Ry(θ2+θ3) matches the desired pose.
    
    The input orientation r is provided as (roll, pitch, yaw) in radians.
    
    :param p: The desired TCP position (x, y, z)
    :param r: The desired TCP orientation (roll, pitch, yaw) in radians.
    :return: A tuple (θ1, θ2, θ3) of joint angles in radians.
    """
    L1 = 0.425
    L2 = 0.39225
    k = 0.01615
    x, y, z = p
    r_roll, r_pitch, r_yaw = r
    temp = x * x + y * y - k * k
    A = math.sqrt(temp) if temp > 0 else 0.0
    theta1 = math.atan2(A * y - k * x, A * x + k * y)
    r_all_sq = x * x + y * y + z * z
    cos_val = (r_all_sq - (L1 * L1 + L2 * L2)) / (2 * L1 * L2)
    cos_val = max(min(cos_val, 1.0), -1.0)
    alpha = math.acos(cos_val)
    C = L1 + L2 * math.cos(alpha)
    theta3_1 = alpha
    D1 = L2 * math.sin(theta3_1)
    theta2_1 = math.atan2(C * A - D1 * z, C * z + D1 * A)
    sum1 = theta2_1 + theta3_1
    theta3_2 = -alpha
    D2 = L2 * math.sin(theta3_2)
    theta2_2 = math.atan2(C * A - D2 * z, C * z + D2 * A)
    sum2 = theta2_2 + theta3_2

    def Rz(angle):
        return [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]

    def Ry(angle):
        return [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]]

    def mat_mult(A, B):
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(3):
            for j in range(3):
                for k_ in range(3):
                    result[i][j] += A[i][k_] * B[k_][j]
        return result

    def frobenius_norm(M):
        s = 0
        for i in range(3):
            for j in range(3):
                s += M[i][j] ** 2
        return math.sqrt(s)
    Rz_t1 = Rz(theta1)
    R_candidate1 = mat_mult(Rz_t1, Ry(sum1))
    R_candidate2 = mat_mult(Rz_t1, Ry(sum2))
    cr = math.cos(r_roll)
    sr = math.sin(r_roll)
    cp = math.cos(r_pitch)
    sp = math.sin(r_pitch)
    cy = math.cos(r_yaw)
    sy = math.sin(r_yaw)
    Rz_r = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    Ry_r = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    Rx_r = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]
    R_temp = mat_mult(Ry_r, Rx_r)
    R_des = mat_mult(Rz_r, R_temp)
    diff1 = [[R_candidate1[i][j] - R_des[i][j] for j in range(3)] for i in range(3)]
    diff2 = [[R_candidate2[i][j] - R_des[i][j] for j in range(3)] for i in range(3)]
    error1 = frobenius_norm(diff1)
    error2 = frobenius_norm(diff2)
    if error1 <= error2:
        chosen_theta3 = theta3_1
        provisional_theta2 = theta2_1
        chosen_sum = sum1
    else:
        chosen_theta3 = theta3_2
        provisional_theta2 = theta2_2
        chosen_sum = sum2

    def transpose(M):
        return [list(row) for row in zip(*M)]

    def wrap(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    Rz_minus_t1 = Rz(-theta1)
    R_temp2 = mat_mult(Rz_minus_t1, R_des)
    delta_eff = math.atan2(-R_temp2[2][0], R_temp2[2][2])
    theta2 = delta_eff - chosen_theta3
    theta1 = wrap(theta1)
    theta2 = wrap(theta2)
    theta3 = wrap(chosen_theta3)
    return (theta1, theta2, theta3)