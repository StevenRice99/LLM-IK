import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    z_adjusted = z - 0.09465
    y_adjusted = y - 0.093
    theta1 = math.atan2(x, z_adjusted)
    r = math.sqrt(x ** 2 + z_adjusted ** 2)
    d = 0.39225
    cos_theta2 = (r ** 2 - d ** 2) / (2 * r * d)
    theta2 = math.acos(cos_theta2)
    theta3 = 0
    return (theta1, theta2, theta3)