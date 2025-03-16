import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    x2 = x - 0.39225 * math.sin(theta1)
    z2 = z - 0.39225 * math.cos(theta1)
    y3 = y - 0.093
    distance = math.sqrt(x2 ** 2 + y3 ** 2 + z2 ** 2)
    if distance > 0.09465 + 0.0823:
        raise ValueError('Target position is not reachable.')
    A = 0.09465
    B = 0.0823
    cos_theta2 = (A ** 2 + B ** 2 - distance ** 2) / (2 * A * B)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    x4 = x2 - 0.09465 * math.sin(theta1 + theta2)
    z4 = z2 - 0.09465 * math.cos(theta1 + theta2)
    theta4 = theta2
    return (theta1, theta2, theta1 + theta2, theta4)