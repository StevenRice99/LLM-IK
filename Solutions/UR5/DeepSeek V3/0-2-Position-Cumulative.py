import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    r = math.sqrt(x ** 2 + y ** 2)
    z_wrist = z - l3
    D = (r ** 2 + z_wrist ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = math.acos(D)
    D = (r ** 2 + z_wrist ** 2 - l1 ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    theta3 = math.acos(D)
    return (theta1, theta2, theta3)