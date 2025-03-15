import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    d = math.sqrt(x ** 2 + y ** 2)
    l1 = 0.13585
    l2 = 0.425
    z_target = z - l1
    d_target = math.sqrt(d ** 2 + z_target ** 2)
    cos_theta3 = (d_target ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_target, d)
    beta = math.asin(l2 * math.sin(theta3) / d_target)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)