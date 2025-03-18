import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    x_prime = x * math.cos(theta1) + z * math.sin(theta1)
    z_prime = -x * math.sin(theta1) + z * math.cos(theta1)
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    d = math.sqrt(x_prime ** 2 + (z_prime - L1) ** 2)
    arg_beta = (L2 ** 2 + d ** 2 - L3 ** 2) / (2 * L2 * d)
    arg_beta = max(min(arg_beta, 1), -1)
    beta = math.acos(arg_beta)
    alpha = math.atan2(x_prime, z_prime - L1)
    theta2 = alpha - beta
    arg_gamma = (L2 ** 2 + L3 ** 2 - d ** 2) / (2 * L2 * L3)
    arg_gamma = max(min(arg_gamma, 1), -1)
    gamma = math.acos(arg_gamma)
    theta3 = math.pi - gamma
    orientation_sum = theta2 + theta3
    orientation_diff = r[1] - orientation_sum
    theta2 += orientation_diff / 2
    theta3 += orientation_diff / 2
    return (theta1, theta2, theta3)