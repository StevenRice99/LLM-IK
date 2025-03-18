import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    z_adj = z - 0.09465
    theta1 = math.atan2(x, z_adj)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z_adj - 0.39225 * math.cos(theta1)
    theta2 = math.atan2(adjusted_x, adjusted_z) - theta1
    x2 = 0.39225 * math.sin(theta1 + theta2)
    y2 = 0.093 * math.cos(theta2)
    dx = x - x2
    dy = y - y2
    theta3 = math.atan2(dx, dy)
    return (theta1, theta2, theta3)