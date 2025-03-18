import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    link_length = math.sqrt((0 - 0) ** 2 + (0 - -0.1197) ** 2 + (0.39225 - 0.425) ** 2)
    numerator = adjusted_x ** 2 + adjusted_z ** 2 - link_length ** 2
    denominator = 2 * adjusted_x * adjusted_z
    theta3 = math.acos(numerator / denominator)
    theta4 = 0.0
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)