import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    d4 = 0.093
    theta1 = np.arctan2(x, z)
    z_adjusted = z - L3 * np.cos(r_y)
    x_adjusted = x - L3 * np.sin(r_y)
    d = np.sqrt(x_adjusted ** 2 + z_adjusted ** 2)
    h = y - d4
    r = np.sqrt(d ** 2 + h ** 2)
    cos_theta3 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    sin_theta3 = np.sqrt(1 - cos_theta3 ** 2)
    theta2 = np.arctan2(h, d) - np.arctan2(L2 * sin_theta3, L1 + L2 * cos_theta3)
    theta4 = r_z - (theta1 + theta2 + theta3)
    return (theta1, theta2, theta3, theta4)