import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    theta1 = math.atan2(y, x)
    x_prime = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    D = math.sqrt(x_prime ** 2 + z_prime ** 2)
    cos_theta3 = (D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    sin_theta3 = math.sqrt(1 - cos_theta3 ** 2)
    theta3 = math.atan2(sin_theta3, cos_theta3)
    alpha = math.atan2(z_prime, x_prime)
    beta = math.atan2(L3 * sin_theta3, L2 + L3 * cos_theta3)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)