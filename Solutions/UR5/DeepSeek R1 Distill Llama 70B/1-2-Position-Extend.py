import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_target, y_target, z_target = p
    l1 = 0.4412
    l2 = 0.39225
    x = x_target
    z = z_target
    K = (x ** 2 + z ** 2 + l2 ** 2 - l1 ** 2) / (2 * l2)
    A = x
    B = z
    C = math.sqrt(A ** 2 + B ** 2)
    if C == 0:
        return (0.0, 0.0)
    phi = math.atan2(B, A)
    theta_candidates = []
    if abs(K / C) <= 1:
        theta1 = phi + math.acos(K / C)
        theta2 = phi - math.acos(K / C)
        theta_candidates.append(theta1)
        theta_candidates.append(theta2)
    solutions = []
    for theta in theta_candidates:
        cos_theta1 = (x - l2 * math.cos(theta)) / l1
        sin_theta1 = (z - l2 * math.sin(theta)) / l1
        if abs(cos_theta1 ** 2 + sin_theta1 ** 2 - 1) > 1e-06:
            continue
        angle1 = math.atan2(sin_theta1, cos_theta1)
        angle2 = theta - angle1
        if -6.2831853 <= angle1 <= 6.2831853 and -6.2831853 <= angle2 <= 6.2831853:
            solutions.append((angle1, angle2))
    if solutions:
        return solutions[0]
    else:
        return (0.0, 0.0)