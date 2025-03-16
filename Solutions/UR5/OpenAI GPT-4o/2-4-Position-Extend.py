import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    r = math.sqrt(x ** 2 + z ** 2)
    theta1 = math.atan2(x, z)
    r_adjusted = r - l3
    d = math.sqrt(r_adjusted ** 2 + y ** 2)
    cos_theta2 = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_theta2 = max(-1, min(1, cos_theta2))
    theta2 = math.acos(cos_theta2)
    theta2_offset = math.atan2(y, r_adjusted)
    theta2 = theta2_offset - theta2
    theta3 = 0
    return (theta1, theta2, theta3)