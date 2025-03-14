import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.093
    l2 = 0.09465
    ratio = l1 / l2
    x_relative = x / l2
    y_relative = y / l2
    theta1 = math.atan2(x_relative, y_relative)
    theta2 = math.atan2(x - l1 * math.sin(theta1), y - l1 * math.cos(theta1))
    return (theta1, theta2)