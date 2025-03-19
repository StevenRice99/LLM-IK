def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    rx, ry, rz = r

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

    def rot_y(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

    def rot_z(angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    l1_y = -0.1197
    l1_z = 0.425
    l2_z = 0.39225
    l3_y = 0.093
    tcp_z = 0.09465
    tcp_offset = np.array([0, 0, tcp_z])
    p_joint4 = np.array([x, y, z]) - R_target @ tcp_offset
    theta1 = math.atan2(p_joint4[0], p_joint4[2])
    R1 = rot_y(theta1)
    p_joint4_local = R1.T @ p_joint4
    p_joint2_local = np.array([0, l1_y, l1_z])
    v = p_joint4_local - p_joint2_local
    L = math.sqrt(v[1] ** 2 + v[2] ** 2)
    L2 = l2_z
    L3 = l3_y
    cos_theta3 = (L ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3_candidates = [math.acos(cos_theta3), -math.acos(cos_theta3)]
    theta2_candidates = []
    for theta3 in theta3_candidates:
        alpha = math.atan2(v[1], v[2])
        beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
        theta2 = alpha - beta
        theta2_candidates.append(theta2)
    theta4_candidates = []
    R_after3_list = []
    for i in range(len(theta3_candidates)):
        theta2 = theta2_candidates[i]
        theta3 = theta3_candidates[i]
        R2 = rot_y(theta2)
        R3 = rot_y(theta3)
        R_after3 = R1 @ R2 @ R3
        R_after3_list.append(R_after3)
        R_needed = R_after3.T @ R_target
        theta4 = math.atan2(R_needed[1, 0], R_needed[0, 0])
        theta4_candidates.append(theta4)
    additional_solutions = []
    for i in range(len(theta3_candidates)):
        theta2 = theta2_candidates[i]
        theta3 = theta3_candidates[i]
        theta4 = theta4_candidates[i]
        additional_solutions.extend([(theta1, theta2, theta3, theta4), (theta1 + 2 * math.pi, theta2, theta3, theta4), (theta1 - 2 * math.pi, theta2, theta3, theta4), (theta1, theta2 + 2 * math.pi, theta3, theta4), (theta1, theta2 - 2 * math.pi, theta3, theta4), (theta1, theta2, theta3 + 2 * math.pi, theta4), (theta1, theta2, theta3 - 2 * math.pi, theta4), (theta1, theta2, theta3, theta4 + 2 * math.pi), (theta1, theta2, theta3, theta4 - 2 * math.pi)])
    best_solution = None
    min_error = float('inf')
    for solution in additional_solutions:
        theta1, theta2, theta3, theta4 = solution
        R1 = rot_y(theta1)
        R2 = rot_y(theta2)
        R3 = rot_y(theta3)
        R4 = rot_z(theta4)
        R_final = R1 @ R2 @ R3 @ R4
        error = np.linalg.norm(R_final - R_target, 'fro')
        if error < min_error:
            min_error = error
            best_solution = solution
    return best_solution