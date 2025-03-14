import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    sin_theta1 = (z - 0.39225) / 0.425
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    theta1 = math.asin(sin_theta1)
    numerator = y + 0.1197 * math.cos(theta1)
    denominator = x - 0.1197 * math.sin(theta1)
    theta2 = math.atan2(numerator, denominator)
    return (theta1, theta2)