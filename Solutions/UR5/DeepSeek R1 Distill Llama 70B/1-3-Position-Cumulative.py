import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
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
    adjusted_x2 = adjusted_x * math.cos(theta2) + adjusted_z * math.sin(theta2)
    adjusted_z2 = -adjusted_x * math.sin(theta2) + adjusted_z * math.cos(theta2)
    theta3 = math.atan2(y, adjusted_x2)
    return (theta1, theta2, theta3)