import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz] (roll, pitch, yaw ZYX).
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    roll, pitch, yaw = r
    d1y = 0.13585
    d2y = -0.1197
    d2z = 0.425
    d3z = 0.39225

    def euler_zyx_to_matrix(y_angle, p_angle, r_angle):
        cy = math.cos(y_angle)
        sy = math.sin(y_angle)
        cp = math.cos(p_angle)
        sp = math.sin(p_angle)
        cr = math.cos(r_angle)
        sr = math.sin(r_angle)
        R = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]
        return R

    def mat_vec_mult(mat, vec):
        res = [0.0, 0.0, 0.0]
        for i in range(3):
            for j_mult in range(3):
                res[i] += mat[i][j_mult] * vec[j_mult]
        return res

    def mat_mat_mult(A, B):
        C = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for i in range(3):
            for j in range(3):
                for k_mult in range(3):
                    C[i][j] += A[i][k_mult] * B[k_mult][j]
        return C
    R_tcp_base = euler_zyx_to_matrix(yaw, pitch, roll)
    V_j3_tcp_local = [0.0, 0.0, d3z]
    offset_vec = mat_vec_mult(R_tcp_base, V_j3_tcp_local)
    pwc_x = px - offset_vec[0]
    pwc_y = py - offset_vec[1]
    pwc_z = pz - offset_vec[2]
    cos_q2_val = pwc_z / d2z
    if cos_q2_val > 1.0:
        cos_q2_val = 1.0
    if cos_q2_val < -1.0:
        cos_q2_val = -1.0
    q2 = math.acos(cos_q2_val)
    s2 = math.sin(q2)
    K_const = d1y + d2y
    A_coeff = d2z * s2
    q1_num = A_coeff * pwc_y - K_const * pwc_x
    q1_den = A_coeff * pwc_x + K_const * pwc_y
    q1 = math.atan2(q1_num, q1_den)
    cos_q1 = math.cos(q1)
    sin_q1 = math.sin(q1)
    Rz_negq1 = [[cos_q1, sin_q1, 0.0], [-sin_q1, cos_q1, 0.0], [0.0, 0.0, 1.0]]
    cos_negq2 = math.cos(-q2)
    sin_negq2 = math.sin(-q2)
    Ry_negq2 = [[cos_negq2, 0.0, sin_negq2], [0.0, 1.0, 0.0], [-sin_negq2, 0.0, cos_negq2]]
    Temp_mat = mat_mat_mult(Rz_negq1, R_tcp_base)
    M = mat_mat_mult(Ry_negq2, Temp_mat)
    q3 = math.atan2(M[0][2], M[0][0])
    return (q1, q2, q3)