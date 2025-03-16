import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    z_adjusted = z - 0.39225
    r = math.sqrt(x ** 2 + y ** 2)
    cos_theta2 = (r ** 2 + z_adjusted ** 2 - 0.425 ** 2 - 0.13585 ** 2) / (2 * 0.425 * 0.13585)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(y, x)
    theta3 = math.atan2(z_adjusted, r)
    return (theta1, theta2, theta3)