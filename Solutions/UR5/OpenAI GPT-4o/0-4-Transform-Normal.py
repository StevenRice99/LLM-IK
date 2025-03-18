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
    d1 = 0.13585
    d2 = 0.425
    d3 = 0.39225
    d4 = 0.093
    d_tcp = 0.09465
    theta1 = np.arctan2(py, px)
    pz_adjusted = pz - d_tcp
    py_adjusted = np.sqrt(px ** 2 + py ** 2) - d4
    L1 = d2
    L2 = d3
    D = (py_adjusted ** 2 + pz_adjusted ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta3 = np.arccos(np.clip(D, -1.0, 1.0))
    phi2 = np.arctan2(pz_adjusted, py_adjusted)
    phi1 = np.arctan2(L2 * np.sin(theta3), L1 + L2 * np.cos(theta3))
    theta2 = phi2 - phi1
    theta4 = ry - (theta2 + theta3)
    theta5 = rz
    return (theta1, theta2, theta3, theta4, theta5)