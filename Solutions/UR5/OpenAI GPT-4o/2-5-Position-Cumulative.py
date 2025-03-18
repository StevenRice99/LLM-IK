import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    theta2 = math.atan2(adjusted_x, adjusted_z) - theta1
    adjusted_y = y - 0.093
    theta3 = math.atan2(adjusted_y, adjusted_x)
    adjusted_z_final = adjusted_z - 0.09465
    theta4 = math.atan2(adjusted_x, adjusted_z_final)
    return (theta1, theta2, theta3, theta4)