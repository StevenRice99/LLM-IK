import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    theta1 = np.arctan2(y, x)
    wx = np.sqrt(x ** 2 + y ** 2)
    wz = z - l1
    d = np.sqrt(wx ** 2 + wz ** 2)
    cos_theta3 = (d ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    alpha = np.arctan2(wz, wx)
    beta = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))
    theta2 = alpha - beta
    theta1 += r_z
    theta2 += r_y
    theta3 += r_x
    return (theta1, theta2, theta3)