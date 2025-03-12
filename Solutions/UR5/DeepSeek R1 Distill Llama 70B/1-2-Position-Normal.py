import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    cos_theta2 = (pz - 0.425) / 0.39225
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    theta1 = math.atan2(-0.39225 * sin_theta2, -0.1197)
    return (theta1, theta2)