import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    theta1 = math.atan2(y, x)
    x_proj = math.sqrt(x ** 2 + y ** 2) - l1
    z_proj = z
    d = math.sqrt(x_proj ** 2 + z_proj ** 2)
    cos_theta3 = (l2 ** 2 + l3 ** 2 - d ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_proj, x_proj)
    beta = math.asin(l3 * math.sin(theta3) / d)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)