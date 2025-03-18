import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L_TCP = 0.093
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    d = np.sqrt(r ** 2 + z_prime ** 2)
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    cos_theta2 = (L2 ** 2 + d ** 2 - L3 ** 2) / (2 * L2 * d)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arctan2(z_prime, r) - np.arccos(cos_theta2)
    theta4 = 0
    return (theta1, theta2, theta3, theta4)