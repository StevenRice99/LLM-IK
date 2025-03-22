def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x_target, y_target, z_target = p
    rx, ry, rz = r
    L1_y = 0.13585
    L2_y = -0.1197
    L2_z = 0.425
    L3_z = 0.39225
    theta1 = rz
    c1 = math.cos(theta1)
    s1 = math.sin(theta1)
    x1_target = c1 * x_target + s1 * y_target
    y1_target = -s1 * x_target + c1 * y_target
    z1_target = z_target
    y1_target -= L1_y
    x2_target = x1_target
    y2_target = y1_target
    z2_target = z1_target
    d_sq = x2_target ** 2 + (y2_target - L2_y) ** 2 + (z2_target - L2_z) ** 2
    d = math.sqrt(d_sq)
    L2_len = math.sqrt(L2_y ** 2 + L2_z ** 2)
    L3_len = L3_z
    cos_theta3 = (d_sq - L2_len ** 2 - L3_len ** 2) / (2 * L2_len * L3_len)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3_1 = math.acos(cos_theta3)
    theta3_2 = -theta3_1
    solutions = []
    for theta3 in [theta3_1, theta3_2]:
        beta = math.atan2(y2_target - L2_y, z2_target - L2_z)
        alpha = math.atan2(L3_len * math.sin(theta3), L2_len + L3_len * math.cos(theta3))
        theta2 = beta - alpha
        sin2 = math.sin(theta2)
        cos2 = math.cos(theta2)
        sin23 = math.sin(theta2 + theta3)
        cos23 = math.cos(theta2 + theta3)
        x_calc = 0
        y_calc = L2_y * cos2 + L3_z * sin23
        z_calc = L2_z * cos2 + L3_z * cos23
        if math.isclose(x_calc, x2_target, abs_tol=0.0001) and math.isclose(y_calc, y2_target, abs_tol=0.0001) and math.isclose(z_calc, z2_target, abs_tol=0.0001):
            expected_ry = theta2 + theta3
            ry_error = min(abs(expected_ry - ry), abs(expected_ry - ry - 2 * math.pi), abs(expected_ry - ry + 2 * math.pi))
            solutions.append((theta2, theta3, ry_error))
    if not solutions:
        for theta3 in [theta3_1, theta3_2]:
            theta2 = ry - theta3
            sin2 = math.sin(theta2)
            cos2 = math.cos(theta2)
            sin23 = math.sin(theta2 + theta3)
            cos23 = math.cos(theta2 + theta3)
            x_calc = 0
            y_calc = L2_y * cos2 + L3_z * sin23
            z_calc = L2_z * cos2 + L3_z * cos23
            pos_error = math.sqrt((x_calc - x2_target) ** 2 + (y_calc - y2_target) ** 2 + (z_calc - z2_target) ** 2)
            solutions.append((theta2, theta3, pos_error))
    solutions.sort(key=lambda s: s[2])
    if solutions:
        theta2, theta3, _ = solutions[0]
    else:
        theta2 = ry
        theta3 = 0
    return (theta1, theta2, theta3)