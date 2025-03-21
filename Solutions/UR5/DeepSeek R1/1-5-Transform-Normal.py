import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    target_pos = np.array(p)
    rx, ry, rz = r
    R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_target = R_z @ R_y @ R_x
    R_adj = R_target @ np.array([[np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0], [np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0], [0, 0, 1]])

    def yzy_from_matrix(R):
        solutions = []
        beta = np.arccos(np.clip(R[1, 1], -1.0, 1.0))
        beta_options = [beta, -beta]
        for beta in beta_options:
            if np.isclose(np.sin(beta), 0):
                alpha = 0.0
                if np.isclose(R[1, 1], 1.0):
                    gamma = np.arctan2(R[0, 2], R[0, 0])
                else:
                    gamma = np.arctan2(-R[0, 2], -R[0, 0])
                solutions.append((alpha, beta, gamma))
                solutions.append((alpha + np.pi, beta, gamma + np.pi))
            else:
                sin_beta = np.sin(beta)
                alpha = np.arctan2(R[2, 1] / sin_beta, -R[0, 1] / sin_beta)
                gamma = np.arctan2(R[1, 2] / sin_beta, R[1, 0] / sin_beta)
                solutions.append((alpha, beta, gamma))
                solutions.append((alpha + np.pi, -beta, gamma + np.pi))
        return solutions
    yzy_solutions = yzy_from_matrix(R_adj)
    L1 = 0.425
    L2 = 0.39225
    wrist_y_world = -0.1197 + 0.093
    required_O_y = target_pos[1] - wrist_y_world
    cos_q4 = required_O_y / 0.0823
    if abs(cos_q4) > 1:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    q4_1 = np.arccos(cos_q4)
    q4_2 = -q4_1
    q4_candidates = [q4_1, q4_2]
    for q4 in q4_candidates:
        offset_x = -0.0823 * np.sin(q4)
        offset_z = 0.09465
        wrist_x = target_pos[0] - offset_x
        wrist_z = target_pos[2] - offset_z
        D = wrist_x ** 2 + wrist_z ** 2
        cos_q2 = (D - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        if abs(cos_q2) > 1:
            continue
        q2_1 = np.arccos(cos_q2)
        q2_2 = -q2_1
        for q2 in [q2_1, q2_2]:
            A = L1 + L2 * np.cos(q2)
            B = L2 * np.sin(q2)
            q1 = np.arctan2(wrist_x, wrist_z) - np.arctan2(B, A)
            for alpha, beta, gamma in yzy_solutions:
                if not np.isclose(beta, q4, atol=1e-06):
                    continue
                theta = alpha
                q3 = theta - q1 - q2
                q5 = gamma
                if all((-6.2831853 <= angle <= 6.2831853 for angle in [q1, q2, q3, q4, q5])):
                    return (float(q1), float(q2), float(q3), float(q4), float(q5))
    return (0.0, 0.0, 0.0, 0.0, 0.0)