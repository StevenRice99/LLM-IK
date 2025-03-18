import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    cos_theta2 = (z - y) / 0.00165
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    sin_theta3 = (y - 0.093 * math.cos(theta2)) / 0.0823
    sin_theta3 = max(min(sin_theta3, 1.0), -1.0)
    theta3 = math.asin(sin_theta3)
    theta1 = math.atan2(x, z)
    return (theta1, theta2, theta3)