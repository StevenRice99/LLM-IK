import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    y_j4 = y - 0.0823
    x_j4 = x
    z_j4 = z
    theta3 = math.atan2(x_j4, y_j4)
    x_j3 = x_j4 - 0.09465 * math.sin(theta3)
    y_j3 = y_j4 - 0.09465 * math.cos(theta3)
    z_j3 = z_j4
    x_j2 = x_j3
    y_j2 = y_j3 - 0.093
    z_j2 = z_j3
    dist_xz = math.sqrt(x_j2 ** 2 + z_j2 ** 2)
    theta1 = math.atan2(x_j2, z_j2)
    x_j2_after_j1 = 0.39225 * math.sin(theta1)
    z_j2_after_j1 = 0.39225 * math.cos(theta1)
    vec_j2_to_j3_x = x_j3 - x_j2_after_j1
    vec_j2_to_j3_z = z_j3 - z_j2_after_j1
    theta2 = math.atan2(vec_j2_to_j3_x, vec_j2_to_j3_z) - theta1
    theta4 = -theta1 - theta2
    return (theta1, theta2, theta3, theta4)