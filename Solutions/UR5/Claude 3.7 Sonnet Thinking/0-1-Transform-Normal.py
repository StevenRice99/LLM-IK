import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    l1 = 0.13585
    l2 = 0.425
    offset_y = -0.1197
    z_adjusted = z - l1
    cos_theta2 = z_adjusted / l2
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2_pos = np.arccos(cos_theta2)
    theta2_neg = -theta2_pos
    solutions = []
    for theta2 in [theta2_pos, theta2_neg]:
        a = l2 * np.sin(theta2)
        b = offset_y
        r_xy = np.sqrt(x * x + y * y)
        phi = np.arctan2(y, x)
        if abs(a) < 1e-10:
            theta1 = phi
        else:
            gamma = np.arctan2(b, a)
            theta1 = phi - gamma
        solutions.append((theta1, theta2))
        solutions.append((theta1 + np.pi, -theta2))
    normalized_solutions = []
    for theta1, theta2 in solutions:
        while theta1 > np.pi:
            theta1 -= 2 * np.pi
        while theta1 < -np.pi:
            theta1 += 2 * np.pi
        while theta2 > np.pi:
            theta2 -= 2 * np.pi
        while theta2 < -np.pi:
            theta2 += 2 * np.pi
        normalized_solutions.append((theta1, theta2))
    best_solution = None
    min_error = float('inf')
    for theta1, theta2 in normalized_solutions:
        rx_error = min(abs(rx), abs(rx - np.pi), abs(rx + np.pi))
        ry_error = min(abs(ry - theta2), abs(ry - theta2 + 2 * np.pi), abs(ry - theta2 - 2 * np.pi))
        rz_error = min(abs(rz - theta1), abs(rz - theta1 + 2 * np.pi), abs(rz - theta1 - 2 * np.pi))
        total_error = rx_error + ry_error + rz_error
        if total_error < min_error:
            min_error = total_error
            best_solution = (theta1, theta2)
    return best_solution