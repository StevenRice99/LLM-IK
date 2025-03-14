import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    y_adjusted = y - 0.093
    A = 0.39225
    B = 0.093
    distance = math.sqrt(x ** 2 + z ** 2)
    theta1 = math.atan2(x, z)
    remaining_distance = distance - A * math.cos(theta1)
    cos_theta2 = (remaining_distance ** 2 - B ** 2) / (2 * A * remaining_distance)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    x_current = A * math.cos(theta1) + B * math.cos(theta1 + theta2)
    z_current = A * math.sin(theta1) + B * math.sin(theta1 + theta2)
    delta_x = x - x_current
    delta_z = z - z_current
    theta3 = math.atan2(delta_x, delta_z)
    return (theta1, theta2, theta3)