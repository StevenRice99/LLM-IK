import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d = -0.1197
    theta1 = np.arctan2(y - d, x)
    r = np.sqrt(x ** 2 + (y - d) ** 2)
    z_prime = z
    cos_theta2 = (r ** 2 + z_prime ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    sin_theta2 = np.sqrt(1 - cos_theta2 ** 2)
    if z_prime < 0:
        sin_theta2 = -sin_theta2
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    return (theta1, theta2)