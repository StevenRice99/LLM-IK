import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    adjusted_x = x
    adjusted_y = y - L2
    adjusted_z = z - L3
    r = math.sqrt(adjusted_x ** 2 + adjusted_z ** 2)
    theta1 = math.atan2(adjusted_x, adjusted_z)
    D = (r ** 2 + adjusted_y ** 2 - L1 ** 2) / (2 * L1 * math.sqrt(r ** 2 + adjusted_y ** 2))
    D = max(min(D, 1.0), -1.0)
    theta2 = math.atan2(adjusted_y, r) - math.acos(D)
    theta3 = 0
    return (theta1, theta2, theta3)