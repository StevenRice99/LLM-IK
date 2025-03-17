import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    sin_theta2 = (z - 0.09465) / 0.0823
    theta2 = math.asin(sin_theta2)
    theta1 = math.atan2(-x, y)
    return (theta1, theta2)