import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    l1 = 0.425
    l2 = 0.39225
    d_tcp = 0.093
    theta_1 = math.atan2(x, z)
    z_adjusted = z - d_tcp * math.cos(pitch)
    x_adjusted = x - d_tcp * math.sin(pitch)
    r = math.sqrt(x_adjusted ** 2 + z_adjusted ** 2)
    cos_theta_2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta_2 = max(-1, min(1, cos_theta_2))
    theta_2 = math.atan2(math.sqrt(1 - cos_theta_2 ** 2), cos_theta_2)
    cos_alpha = (l1 ** 2 + r ** 2 - l2 ** 2) / (2 * l1 * r)
    cos_alpha = max(-1, min(1, cos_alpha))
    alpha = math.atan2(math.sqrt(1 - cos_alpha ** 2), cos_alpha)
    beta = math.atan2(z_adjusted, x_adjusted)
    theta_3 = beta - alpha
    return (theta_1, theta_2, theta_3)