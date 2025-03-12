import numpy as np
import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - 0.39225
    L1 = 0.13585
    L2 = math.sqrt((-0.1197) ** 2 + 0.425 ** 2)
    D = (r ** 2 + z_prime ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    D = max(-1.0, min(D, 1.0))
    theta3 = math.atan2(math.sqrt(1 - D ** 2), D)
    theta2 = math.atan2(z_prime, r) - math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    return (theta1, theta2, theta3)