import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1, theta2 = inverse_kinematics_joints_1_to_2((x, y, z))
    x2 = 0.39225 * math.sin(theta1) + 0.093 * math.sin(theta1 + theta2)
    y2 = 0.39225 * math.cos(theta1) + 0.093 * math.cos(theta1 + theta2)
    z2 = 0.0
    x_tcp = x - x2
    y_tcp = y - y2
    z_tcp = z - z2
    theta3, theta4 = inverse_kinematics_joints_3_to_4((x_tcp, y_tcp, z_tcp))
    return (theta1, theta2, theta3, theta4)

def inverse_kinematics_joints_1_to_2(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" for joints 1 and 2.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x - 0.39225 * math.sin(theta1)
    adjusted_z = z - 0.39225 * math.cos(theta1)
    theta_total = math.atan2(adjusted_x, adjusted_z)
    theta2 = theta_total - theta1
    return (theta1, theta2)

def inverse_kinematics_joints_3_to_4(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" for joints 3 and 4.
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta3 = math.atan2(-x, y)
    theta4 = 0.0
    return (theta3, theta4)