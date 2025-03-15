import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    d = math.sqrt(x_target ** 2 + z_target ** 2)
    if d > L1 + L2:
        raise ValueError('Target is out of reach')
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta2 = math.atan2(sin_theta2, cos_theta2)
    alpha = math.atan2(z_target, x_target)
    beta = math.atan2(L2 * sin_theta2, L1 + L2 * cos_theta2)
    theta1 = alpha - beta
    if x_target < 0:
        theta1 = math.pi - theta1
        theta2 = -theta2
    return (theta1, theta2)