import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.093
    L2 = 0.09465
    L3 = 0.0823
    theta1 = np.arctan2(x, -z)
    r = np.sqrt(x ** 2 + z ** 2)
    d = y - L3
    D = (r ** 2 + d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    D = np.clip(D, -1.0, 1.0)
    theta2 = np.arctan2(d, r) - np.arccos(D)
    theta3 = 0
    return (theta1, theta2, theta3)