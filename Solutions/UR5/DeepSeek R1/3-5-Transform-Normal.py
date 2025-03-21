import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    rx_target, ry_target, rz_target = r
    y_offset = y_target - 0.093
    if abs(y_offset) > 0.0823:
        raise ValueError('Target y is out of reach')
    q2 = np.arccos(y_offset / 0.0823)
    q2_solutions = [q2, -q2]
    solutions = []
    for q2 in q2_solutions:
        A = -0.0823 * np.sin(q2)
        B = 0.09465
        denominator = A ** 2 + B ** 2
        if np.isclose(denominator, 0):
            continue
        cos_q1_num = A * x_target + B * z_target
        sin_q1_num = B * x_target - A * z_target
        cos_q1 = cos_q1_num / denominator
        sin_q1 = sin_q1_num / denominator
        q1 = np.arctan2(sin_q1, cos_q1)
        cx, sx = (np.cos(rx_target), np.sin(rx_target))
        cy, sy = (np.cos(ry_target), np.sin(ry_target))
        cz, sz = (np.cos(rz_target), np.sin(rz_target))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R_target = Rz @ Ry @ Rx
        R_Yq1 = np.array([[np.cos(q1), 0, np.sin(q1)], [0, 1, 0], [-np.sin(q1), 0, np.cos(q1)]])
        R_Zq2 = np.array([[np.cos(q2), -np.sin(q2), 0], [np.sin(q2), np.cos(q2), 0], [0, 0, 1]])
        R_z_pi2_inv = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        R_required = R_Yq1.T @ R_Zq2.T @ R_target @ R_z_pi2_inv
        cos_q3 = R_required[0, 0]
        sin_q3 = R_required[0, 2]
        q3 = np.arctan2(sin_q3, cos_q3)
        solutions.append((q1, q2, q3))
    valid_solutions = [sol for sol in solutions if all((-6.2831853 <= angle <= 6.2831853 for angle in sol))]
    if valid_solutions:
        return min(valid_solutions, key=lambda x: sum(abs(np.array(x))))
    return (0.0, 0.0, 0.0)