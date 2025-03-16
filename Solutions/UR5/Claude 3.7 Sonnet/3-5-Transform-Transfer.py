def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
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
    A = 0.09465
    B = 0.0823
    L2 = 0.093

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def rot_y(angle):
        return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def rot_z(angle):
        return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R_target = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    R_tcp_offset = rot_z(1.570796325)
    R_link3 = R_target @ np.linalg.inv(R_tcp_offset)
    cosθ2 = (y - L2) / B
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2_sol1 = math.acos(cosθ2)
    θ2_sol2 = -θ2_sol1
    best_solution = None
    min_error = float('inf')
    for θ2 in [θ2_sol1, θ2_sol2]:
        sinθ2 = math.sin(θ2)
        M = [[A, -B * sinθ2], [B * sinθ2, A]]
        det_M = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        inv_det_M = 1.0 / det_M
        inv_M = [[inv_det_M * M[1][1], -inv_det_M * M[0][1]], [-inv_det_M * M[1][0], inv_det_M * M[0][0]]]
        sinθ1 = inv_M[0][0] * x + inv_M[0][1] * z
        cosθ1 = inv_M[1][0] * x + inv_M[1][1] * z
        sinθ1 = max(min(sinθ1, 1.0), -1.0)
        cosθ1 = max(min(cosθ1, 1.0), -1.0)
        θ1 = math.atan2(sinθ1, cosθ1)
        R1 = rot_y(θ1)
        R2 = rot_z(θ2)
        R12 = R1 @ R2
        R3_required = np.linalg.inv(R12) @ R_link3
        cosθ3 = R3_required[0, 0]
        sinθ3 = R3_required[0, 2]
        cosθ3 = max(min(cosθ3, 1.0), -1.0)
        sinθ3 = max(min(sinθ3, 1.0), -1.0)
        θ3 = math.atan2(sinθ3, cosθ3)
        for θ3_offset in [0, math.pi]:
            θ3_test = θ3 + θ3_offset
            if θ3_test > math.pi:
                θ3_test -= 2 * math.pi
            elif θ3_test < -math.pi:
                θ3_test += 2 * math.pi
            R3 = rot_y(θ3_test)
            R_result = R1 @ R2 @ R3 @ R_tcp_offset
            error = np.sum(np.abs(R_result - R_target))
            if error < min_error:
                min_error = error
                best_solution = (θ1, θ2, θ3_test)
    return best_solution