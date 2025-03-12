import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    L1 = 0.39225
    L2 = 0.093
    x, y, z = p
    D_squared = x ** 2 + z ** 2
    D = math.sqrt(D_squared)
    numerator = D_squared - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_q2 = numerator / denominator
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2 = math.acos(cos_q2)
    theta = math.atan2(x, z)
    q1 = theta - q2
    return (q1, q2)