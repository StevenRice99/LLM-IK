import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    z_eff = z - l3
    d = np.sqrt(r ** 2 + (z_eff - l1) ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    alpha = np.arctan2(z_eff - l1, r)
    beta = np.arcsin(l3 * np.sin(theta3) / d)
    theta2 = alpha - beta
    return (theta1, theta2, theta3)