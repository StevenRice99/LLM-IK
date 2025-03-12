import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    sin_theta1 = x / 0.093
    cos_theta1 = (z - 0.09465) / 0.093
    sin_theta1 = max(min(sin_theta1, 1.0), -1.0)
    cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
    theta1 = math.atan2(sin_theta1, cos_theta1)
    theta2 = 0.0
    return (theta1, theta2)