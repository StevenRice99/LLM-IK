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
    numerator = A ** 2 + x ** 2 + y ** 2 - B ** 2 - z ** 2
    denominator = 2 * A * x
    if denominator == 0:
        sin_theta1 = 0.0
    else:
        sin_theta1 = numerator / denominator
    cos_theta1 = z / A
    theta1 = math.atan2(sin_theta1, cos_theta1)
    cos_theta2 = y / B
    theta2 = math.acos(cos_theta2)
    return (theta1, theta2)