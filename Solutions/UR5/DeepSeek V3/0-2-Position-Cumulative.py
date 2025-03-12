import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    dx = x - 0
    dy = y - 0.13585
    dz = z - 0
    d2 = 0.425
    d3 = 0.39225
    D = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    cos_theta2 = (D ** 2 - d2 ** 2 - d3 ** 2) / (2 * d2 * d3)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2)
    cos_theta3 = (d2 ** 2 + d3 ** 2 - D ** 2) / (2 * d2 * d3)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)
    return (theta1, theta2, theta3)