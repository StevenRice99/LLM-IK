import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    R = math.sqrt(x ** 2 + z ** 2)
    if R < abs(L1 - L2) or R > L1 + L2:
        return (0.0, 0.0)
    A = x ** 2 + z ** 2 + L1 ** 2 - L2 ** 2
    c = A / (2 * L1 * R)
    if abs(c) > 1.0:
        return (0.0, 0.0)
    phi = math.atan2(z, x)
    theta1 = phi + math.acos(c)
    theta2 = phi - math.acos(c)
    solutions = []
    for theta in [theta1, theta2]:
        cos_q1 = (x - L2 * math.cos(theta)) / L1
        sin_q1 = (z - L2 * math.sin(theta)) / L1
        if abs(cos_q1 ** 2 + sin_q1 ** 2 - 1.0) > 1e-06:
            continue
        q1 = math.atan2(sin_q1, cos_q1)
        q2 = theta - q1
        solutions.append((q1, q2))
    if solutions:
        return solutions[0]
    else:
        return (0.0, 0.0)