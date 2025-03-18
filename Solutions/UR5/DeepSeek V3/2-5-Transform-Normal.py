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
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    theta1 = np.arctan2(x, z)
    d = np.sqrt(x ** 2 + z ** 2)
    h = y - L4 * np.sin(r_y)
    D = (d ** 2 + h ** 2 - L1 ** 2 - L2 ** 2 - L3 ** 2) / (2 * L1 * np.sqrt(L2 ** 2 + L3 ** 2))
    theta2 = np.arctan2(np.sqrt(1 - D ** 2), D) - np.arctan2(L3, L2)
    theta3 = r_z
    theta4 = r_y - theta2
    return (theta1, theta2, theta3, theta4)