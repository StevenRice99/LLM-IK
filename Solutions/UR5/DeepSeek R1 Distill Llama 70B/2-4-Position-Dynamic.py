import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    cos_theta2 = (z - 0.09465) / 0.39225
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    A = 0.39225 * sin_theta2
    B = 0.186
    denominator = A ** 2 + B ** 2
    if denominator == 0:
        return (0.0, 0.0, 0.0)
    cos_theta3 = (A * x + B * y) / denominator
    sin_theta3 = (-B * x + A * y) / denominator
    theta3 = math.atan2(sin_theta3, cos_theta3)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    theta1 = 0.0
    return (theta1, theta2, theta3)