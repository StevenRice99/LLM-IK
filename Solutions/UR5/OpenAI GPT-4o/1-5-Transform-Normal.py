import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.09465
    d4 = 0.0823
    wx = px
    wy = py - d4 * np.cos(rz)
    wz = pz - d4 * np.sin(rz)
    theta1 = np.arctan2(wy, wx)
    r = np.sqrt(wx ** 2 + wy ** 2)
    s = wz - d1
    D = (r ** 2 + s ** 2 - d2 ** 2 - d3 ** 2) / (2 * d2 * d3)
    D = max(min(D, 1.0), -1.0)
    theta3 = np.arctan2(np.sqrt(1 - D ** 2), D)
    phi2 = np.arctan2(s, r)
    phi1 = np.arctan2(d3 * np.sin(theta3), d2 + d3 * np.cos(theta3))
    theta2 = phi2 - phi1
    theta4 = rx
    theta5 = ry
    return (theta1, theta2, theta3, theta4, theta5)