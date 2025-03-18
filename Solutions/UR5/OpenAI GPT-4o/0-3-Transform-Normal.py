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
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L_tcp = 0.093
    theta1 = np.arctan2(y, x)
    d_xy = np.sqrt(x ** 2 + y ** 2) - L_tcp
    if d_xy < 0:
        raise ValueError('Target is too close to the base.')
    d = np.sqrt(d_xy ** 2 + (z - L1) ** 2)
    h = z - L1
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    if not -1 <= cos_theta3 <= 1:
        raise ValueError('Target is unreachable due to joint constraints.')
    theta3 = np.arccos(cos_theta3)
    phi2 = np.arctan2(h, d_xy)
    phi1 = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = phi2 - phi1
    theta4 = r_y - (theta2 + theta3)
    return (theta1, theta2, theta3, theta4)