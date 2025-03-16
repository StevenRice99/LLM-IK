def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    x_target = p[0]
    y_target = p[1]
    z_target = p[2]
    x_rot = r[0]
    y_rot = r[1]
    z_rot = r[2]
    l1 = 0.425
    l2 = 0.39225
    y_offset = -0.1197
    flip_needed = abs(x_rot - math.pi) < 1e-06 and abs(z_rot - math.pi) < 1e-06
    r_xy = math.sqrt(x_target ** 2 + z_target ** 2)
    cos_theta2 = (r_xy ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_pos = math.acos(cos_theta2)
    theta2_neg = -theta2_pos
    phi = math.atan2(x_target, z_target)
    psi_pos = math.atan2(l2 * math.sin(theta2_pos), l1 + l2 * math.cos(theta2_pos))
    theta1_pos = phi - psi_pos
    psi_neg = math.atan2(l2 * math.sin(theta2_neg), l1 + l2 * math.cos(theta2_neg))
    theta1_neg = phi - psi_neg
    solutions = []
    solutions.append((theta1_pos, theta2_pos))
    solutions.append((theta1_neg, theta2_neg))
    solutions.append((theta1_pos + math.pi, theta2_pos))
    solutions.append((theta1_neg + math.pi, theta2_neg))
    solutions.append((theta1_pos, theta2_pos + 2 * math.pi))
    solutions.append((theta1_neg, theta2_neg + 2 * math.pi))
    solutions.append((theta1_pos + math.pi, theta2_pos + 2 * math.pi))
    solutions.append((theta1_neg + math.pi, theta2_neg + 2 * math.pi))
    solutions.append((theta1_pos, theta2_pos - 2 * math.pi))
    solutions.append((theta1_neg, theta2_neg - 2 * math.pi))
    solutions.append((theta1_pos + math.pi, theta2_pos - 2 * math.pi))
    solutions.append((theta1_neg + math.pi, theta2_neg - 2 * math.pi))
    best_solution = None
    min_error = float('inf')
    for s in solutions:
        theta1, theta2 = s
        y_orient = theta1 + theta2
        if flip_needed:
            error = abs(y_orient - y_rot)
        else:
            error = abs(y_orient - y_rot)
        if error < min_error:
            min_error = error
            best_solution = s
    return best_solution