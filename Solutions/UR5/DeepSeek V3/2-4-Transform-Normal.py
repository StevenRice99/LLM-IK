import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    roll, pitch, yaw = r
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    theta3 = yaw
    d = np.sqrt(x ** 2 + y ** 2)
    h = z - L1
    cos_theta2 = (d ** 2 + h ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    alpha = np.arctan2(h, d)
    beta = np.arctan2(L3 * np.sin(theta2), L2 + L3 * np.cos(theta2))
    theta1 = alpha - beta
    return (theta1, theta2, theta3)