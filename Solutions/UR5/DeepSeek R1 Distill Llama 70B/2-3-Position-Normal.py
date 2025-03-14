import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.39225
    B = 0.093
    distance = math.sqrt(x ** 2 + z ** 2)
    cos_q2 = (A ** 2 + B ** 2 - distance ** 2) / (2 * A * B)
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    sin_q2 = math.sqrt(1 - cos_q2 ** 2)
    q2 = math.atan2(sin_q2, cos_q2)
    cos_q1 = (A ** 2 + distance ** 2 - B ** 2) / (2 * A * distance)
    cos_q1 = max(min(cos_q1, 1.0), -1.0)
    sin_q1 = math.sqrt(1 - cos_q1 ** 2)
    q1 = math.atan2(sin_q1, cos_q1)
    return (q1, q2)