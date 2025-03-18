import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    z_prime = z - 0.093
    theta_1 = math.atan2(x, z_prime)
    link_length = 0.39225
    cos_theta_2 = z_prime / link_length
    cos_theta_2 = max(-1.0, min(1.0, cos_theta_2))
    theta_2 = math.acos(cos_theta_2)
    return (theta_1, theta_2)