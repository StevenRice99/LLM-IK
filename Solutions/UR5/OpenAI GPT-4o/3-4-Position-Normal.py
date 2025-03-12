import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    r = math.sqrt(x ** 2 + (z - 0.09465) ** 2)
    cos_theta_1 = (z - 0.09465) / r
    sin_theta_1 = x / r
    theta_1 = math.atan2(sin_theta_1, cos_theta_1)
    theta_2 = math.atan2(y - 0.093, x)
    return (theta_1, theta_2)