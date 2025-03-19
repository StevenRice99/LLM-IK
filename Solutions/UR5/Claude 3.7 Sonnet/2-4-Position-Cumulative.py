import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    L1 = 0.39225
    x2 = x - L1 * math.sin(theta1)
    z2 = z - L1 * math.cos(theta1)
    L3 = 0.09465
    theta_total = math.atan2(x2, z2)
    theta2 = theta_total - theta1
    theta3 = math.atan2(y, x)
    return (theta1, theta2, theta3)