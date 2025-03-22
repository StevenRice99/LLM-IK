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
    roll, pitch, yaw = r
    j1_to_j2 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    TCP_OFFSET = 0.093
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])
    tcp_offset_world = R @ np.array([0, TCP_OFFSET, 0])
    wx = x - tcp_offset_world[0]
    wy = y - tcp_offset_world[1]
    wz = z - tcp_offset_world[2]
    theta1 = math.atan2(wx, wy)
    c1, s1 = (math.cos(theta1), math.sin(theta1))
    x_local = c1 * wx + s1 * wy
    y_local = -s1 * wx + c1 * wy - j1_to_j2
    z_local = wz
    r_xz = math.sqrt(x_local ** 2 + z_local ** 2)
    cos_theta3 = (r_xz ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3_up = math.acos(cos_theta3)
    theta3_down = -theta3_up
    R1 = np.array([[c1, s1, 0], [-s1, c1, 0], [0, 0, 1]])
    R_local = R1.T @ R
    y_rotation = math.atan2(R_local[0, 2], R_local[0, 0])
    solutions = []
    for theta3_candidate in [theta3_up, theta3_down]:
        beta = math.atan2(x_local, z_local)
        alpha = math.atan2(L3 * math.sin(theta3_candidate), L2 + L3 * math.cos(theta3_candidate))
        theta2 = beta - alpha
        theta4 = y_rotation - theta2 - theta3_candidate
        theta4 = (theta4 + math.pi) % (2 * math.pi) - math.pi
        solutions.append((theta1, theta2, theta3_candidate, theta4))
    best_solution = None
    min_error = float('inf')
    for sol in solutions:
        t1, t2, t3, t4 = sol
        R1 = np.array([[math.cos(t1), math.sin(t1), 0], [-math.sin(t1), math.cos(t1), 0], [0, 0, 1]])
        R2 = np.array([[math.cos(t2), 0, math.sin(t2)], [0, 1, 0], [-math.sin(t2), 0, math.cos(t2)]])
        R3 = np.array([[math.cos(t3), 0, math.sin(t3)], [0, 1, 0], [-math.sin(t3), 0, math.cos(t3)]])
        R4 = np.array([[math.cos(t4), 0, math.sin(t4)], [0, 1, 0], [-math.sin(t4), 0, math.cos(t4)]])
        R_combined = R1 @ R2 @ R3 @ R4
        error = np.sum(np.abs(R_combined - R))
        if error < min_error:
            min_error = error
            best_solution = sol
    return best_solution