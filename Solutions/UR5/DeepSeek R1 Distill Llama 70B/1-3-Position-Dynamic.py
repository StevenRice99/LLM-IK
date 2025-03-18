import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    assert x == 0, 'x must be zero'
    theta1 = math.atan2(x, z)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta2 = theta_total - theta1
    alpha = theta1 + theta2
    y0 = -0.1197 + 0.39225 * math.cos(alpha)
    z0 = 0.425 + 0.39225 * math.sin(alpha)
    delta_y = y - y0
    delta_z = z - z0
    phi = math.atan2(delta_z, delta_y)
    theta3 = phi - alpha
    return (theta1, theta2, theta3)