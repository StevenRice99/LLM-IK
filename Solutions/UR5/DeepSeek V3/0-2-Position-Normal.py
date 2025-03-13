import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    theta1 = math.atan2(y, x)
    x_prime = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    d = math.sqrt(x_prime ** 2 + z_prime ** 2)
    arg = (L2 ** 2 + L3 ** 2 - d ** 2) / (2 * L2 * L3)
    arg = max(-1.0, min(1.0, arg))
    theta3 = math.acos(arg)
    alpha = math.atan2(z_prime, x_prime)
    beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)