import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.13585
    B = 0.1197
    a = 0.425
    b = 0.39225
    q1 = math.atan2(-x, y + B)
    C = x + A * math.sin(q1)
    D = z
    numerator = C ** 2 + D ** 2 - a ** 2 - b ** 2
    denominator = 2 * a * b
    cos_phi = numerator / denominator
    cos_phi = max(min(cos_phi, 1.0), -1.0)
    phi = math.acos(cos_phi)
    K = a + b * cos_phi
    L = b * math.sin(phi)
    M = math.sqrt(K ** 2 + L ** 2)
    if M == 0:
        M = 1e-09
    theta = math.atan2(L, K)
    argument = D / M
    argument = max(min(argument, 1.0), -1.0)
    delta = math.acos(argument)
    q2 = theta + delta
    q3 = q2 - phi
    return (q1, q2, q3)