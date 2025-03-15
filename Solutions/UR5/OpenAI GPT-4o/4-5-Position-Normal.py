import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.09465
    l2 = 0.0823
    theta1 = math.atan2(y, x)
    sin_theta2 = (z - d1) / l2
    sin_theta2 = max(-1.0, min(1.0, sin_theta2))
    theta2 = math.asin(sin_theta2)
    return (theta1, theta2)