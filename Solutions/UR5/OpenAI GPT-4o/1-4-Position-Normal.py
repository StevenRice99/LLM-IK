import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    d_y = 0.093
    y_eff = y - d_y
    theta1 = np.arctan2(x, -y_eff)
    r = np.sqrt(x ** 2 + y_eff ** 2)
    z_eff = z - L3
    d = np.sqrt(r ** 2 + z_eff ** 2)
    if d > L1 + L2 or d < abs(L1 - L2):
        raise ValueError('Target position is out of reach')
    cos_theta3 = (L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    cos_theta2 = (L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d)
    theta2 = np.arctan2(z_eff, r) - np.arccos(np.clip(cos_theta2, -1.0, 1.0))
    theta4 = 0
    return (theta1, theta2, theta3, theta4)