import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    A = 0.093
    B = 0.09465
    C = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (A ** 2 + C ** 2 - B ** 2) / (2 * A * C)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    cos_theta3 = (A ** 2 + B ** 2 - C ** 2) / (2 * A * B)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.acos(cos_theta3)
    return (theta1, theta2, theta3)