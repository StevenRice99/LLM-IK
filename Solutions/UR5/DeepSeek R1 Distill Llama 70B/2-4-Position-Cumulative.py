import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_y = y - 0.093
    link_length = 0.09465
    theta2 = math.acos(adjusted_y / link_length)
    adjusted_x = x * math.cos(theta1) + z * math.sin(theta1)
    adjusted_y = y * math.cos(theta1) - z * math.sin(theta1)
    theta3 = math.atan2(adjusted_y, adjusted_x)
    return (theta1, theta2, theta3)