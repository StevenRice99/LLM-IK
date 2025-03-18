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
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta2 = theta_total - theta1
    joint3_x = 0.39225 * math.sin(theta1) + 0.093 * math.sin(theta1 + theta2)
    joint3_y = 0.093 * math.cos(theta1 + theta2)
    joint3_z = 0.39225 * math.cos(theta1) + 0.093 * math.cos(theta1 + theta2)
    dx = x - joint3_x
    dy = y - joint3_y
    dz = z - joint3_z
    local_x = math.cos(theta1 + theta2)
    local_y = math.sin(theta1 + theta2)
    theta3 = math.atan2(dy, dx) - math.atan2(local_y, local_x)
    theta3 = (theta3 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2, theta3)