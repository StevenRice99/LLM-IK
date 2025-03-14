import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x_t, y_t, z_t = p
    A = 0.39225
    B = 0.093
    C = 0.09465
    target_distance = math.sqrt(x_t ** 2 + z_t ** 2)
    cos_theta2 = (A ** 2 + B ** 2 - target_distance ** 2) / (2 * A * B)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(z_t, x_t)
    theta3 = 0.0
    return (theta1, theta2, theta3)