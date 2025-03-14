import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    q1 = math.atan2(y, x)
    A = x * math.cos(q1) + y * math.sin(q1)
    C = 0.39225
    D = 0.425
    numerator = A ** 2 + z ** 2 - C ** 2 - D ** 2
    denominator = 2 * C * D
    cos_q3 = numerator / denominator
    cos_q3 = max(min(cos_q3, 1.0), -1.0)
    q3 = math.acos(cos_q3)
    E = C * math.cos(q3) + D
    F = -C * math.sin(q3)
    phi = math.atan2(F, E)
    magnitude = math.sqrt(E ** 2 + F ** 2)
    if magnitude < 1e-09:
        q2 = 0.0
    else:
        q2 = phi + math.acos(z / magnitude)
    return (q1, q2, q3)