import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    d1 = -0.1197
    sin_theta2 = (z - d1) / l2
    if sin_theta2 < -1:
        sin_theta2 = -1
    elif sin_theta2 > 1:
        sin_theta2 = 1
    theta2 = math.asin(sin_theta2)
    theta1 = math.atan2(y, x)
    return (theta1, theta2)