import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the 6 joint values needed to reach the position.
    """
    x, y, z = p
    L1_y = 0.13585
    L2_y = -0.1197
    L2_z = 0.425
    L3_z = 0.39225
    L4_y = 0.093
    L5_z = 0.09465
    L6_y = 0.0823

    def forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6):
        c1, s1 = (math.cos(theta1), math.sin(theta1))
        R1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        c2, s2 = (math.cos(theta2), math.sin(theta2))
        R2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        c3, s3 = (math.cos(theta3), math.sin(theta3))
        R3 = np.array([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]])
        c4, s4 = (math.cos(theta4), math.sin(theta4))
        R4 = np.array([[c4, 0, s4], [0, 1, 0], [-s4, 0, c4]])
        c5, s5 = (math.cos(theta5), math.sin(theta5))
        R5 = np.array([[c5, -s5, 0], [s5, c5, 0], [0, 0, 1]])
        c6, s6 = (math.cos(theta6), math.sin(theta6))
        R6 = np.array([[c6, 0, s6], [0, 1, 0], [-s6, 0, c6]])
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T2 = np.eye(4)
        T2[:3, :3] = R2
        T2[1, 3] = L1_y
        T3 = np.eye(4)
        T3[:3, :3] = R3
        T3[1, 3] = L2_y
        T3[2, 3] = L2_z
        T4 = np.eye(4)
        T4[:3, :3] = R4
        T4[2, 3] = L3_z
        T5 = np.eye(4)
        T5[:3, :3] = R5
        T5[1, 3] = L4_y
        T6 = np.eye(4)
        T6[:3, :3] = R6
        T6[2, 3] = L5_z
        T_tcp = np.eye(4)
        T_tcp[1, 3] = L6_y
        T = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ T_tcp
        return T[:3, 3]
    best_solution = None
    min_error = float('inf')
    for theta1_sign in [-1, 1]:
        if abs(x) < 1e-10 and abs(y) < 1e-10:
            theta1_candidates = [0.0]
        else:
            base_theta1 = math.atan2(x, y)
            theta1_candidates = [base_theta1, base_theta1 + math.pi, base_theta1 - math.pi]
        for theta1 in theta1_candidates:
            for theta5_approx in [0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]:
                c1, s1 = (math.cos(theta1), math.sin(theta1))
                c5, s5 = (math.cos(theta5_approx), math.sin(theta5_approx))
                x_offset = -L6_y * s1 * c5
                y_offset = L6_y * c1 * c5
                z_offset = L5_z
                wc_x = x - x_offset
                wc_y = y - y_offset
                wc_z = z - z_offset
                wc_x_2 = c1 * wc_x + s1 * wc_y
                wc_y_2 = -s1 * wc_x + c1 * wc_y - L1_y
                wc_z_2 = wc_z
                wc_dist = math.sqrt(wc_x_2 ** 2 + wc_y_2 ** 2 + wc_z_2 ** 2)
                L2_eff = math.sqrt(L2_y ** 2 + L2_z ** 2)
                L3_eff = math.sqrt(L3_z ** 2 + L4_y ** 2)
                cos_theta3 = (wc_dist ** 2 - L2_eff ** 2 - L3_eff ** 2) / (2 * L2_eff * L3_eff)
                if abs(cos_theta3) > 1.0:
                    continue
                for theta3_sign in [-1, 1]:
                    theta3_inner = math.acos(max(min(cos_theta3, 1.0), -1.0))
                    theta3 = theta3_sign * theta3_inner
                    alpha2 = math.atan2(L2_z, -L2_y)
                    beta = math.atan2(L3_z, L4_y) if L4_y != 0 else math.pi / 2
                    cos_gamma = (L2_eff ** 2 + wc_dist ** 2 - L3_eff ** 2) / (2 * L2_eff * wc_dist)
                    cos_gamma = max(min(cos_gamma, 1.0), -1.0)
                    gamma = math.acos(cos_gamma)
                    phi = math.atan2(wc_z_2, math.sqrt(wc_x_2 ** 2 + wc_y_2 ** 2))
                    delta = math.atan2(wc_y_2, wc_x_2)
                    for theta2_sign in [-1, 1]:
                        theta2 = delta - gamma + alpha2
                        for theta4_offset in [0, math.pi, -math.pi, 2 * math.pi, -2 * math.pi]:
                            theta4 = -(theta2 + theta3) + theta4_offset
                            for theta6 in [0]:
                                tcp_calc = forward_kinematics(theta1, theta2, theta3, theta4, theta5_approx, theta6)
                                error = math.sqrt((tcp_calc[0] - x) ** 2 + (tcp_calc[1] - y) ** 2 + (tcp_calc[2] - z) ** 2)
                                if error < min_error:
                                    min_error = error
                                    best_solution = (theta1, theta2, theta3, theta4, theta5_approx, theta6)
    if min_error > 0.1:
        theta1 = math.atan2(x, y)
        projected_dist = math.sqrt(x ** 2 + y ** 2)
        elevation = math.atan2(z, projected_dist)
        for bend_factor in [-0.5, 0, 0.5, 1.0, -1.0]:
            theta2 = elevation + bend_factor * math.pi / 2
            theta3 = -bend_factor * math.pi / 2
            theta4 = bend_factor * math.pi / 2
            for theta5 in [0, math.pi / 2, -math.pi / 2]:
                theta6 = 0
                tcp_calc = forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6)
                error = math.sqrt((tcp_calc[0] - x) ** 2 + (tcp_calc[1] - y) ** 2 + (tcp_calc[2] - z) ** 2)
                if error < min_error:
                    min_error = error
                    best_solution = (theta1, theta2, theta3, theta4, theta5, theta6)
    if best_solution is None:
        return (0, 0, 0, 0, 0, 0)

    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    normalized_solution = tuple((normalize_angle(angle) for angle in best_solution))
    return normalized_solution