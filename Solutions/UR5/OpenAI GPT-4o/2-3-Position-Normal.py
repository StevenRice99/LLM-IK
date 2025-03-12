import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l2 = 0.39225
    offset_y = 0.093
    y_adjusted = y - offset_y
    theta2 = math.atan2(-z, y_adjusted)
    r = math.sqrt(x ** 2 + z ** 2)
    theta1 = math.atan2(x, -z)
    return (theta1, theta2)