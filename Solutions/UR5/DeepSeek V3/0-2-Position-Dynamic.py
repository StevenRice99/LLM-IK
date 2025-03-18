import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    x_transformed = x * cos_theta1 + y * sin_theta1
    y_transformed = -x * sin_theta1 + y * cos_theta1 - 0.13585
    z_transformed = z
    theta2, theta3 = inverse_kinematics_joints_2_3((x_transformed, y_transformed, z_transformed))
    return (theta1, theta2, theta3)

def inverse_kinematics_joints_2_3(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" for the second and third joints.
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if cos_theta2 < -1:
        cos_theta2 = -1
    elif cos_theta2 > 1:
        cos_theta2 = 1
    theta2 = math.acos(cos_theta2)
    cross_product = x * (L1 + L2 * math.cos(theta2)) - z * (L2 * math.sin(theta2))
    if cross_product < 0:
        theta2 = -theta2
    alpha = math.atan2(z, x)
    beta = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta
    return (theta1, theta2)