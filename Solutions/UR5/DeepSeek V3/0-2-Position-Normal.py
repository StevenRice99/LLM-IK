import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    joint1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    s = z - 0.13585
    l1 = 0.425
    l2 = 0.39225
    D = (r ** 2 + s ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if D > 1 or D < -1:
        return (np.nan, np.nan, np.nan)
    joint3 = np.arctan2(np.sqrt(1 - D ** 2), D)
    alpha = np.arctan2(s, r)
    beta = np.arctan2(l2 * np.sin(joint3), l1 + l2 * np.cos(joint3))
    joint2 = alpha - beta
    joint3_alt = -joint3
    beta_alt = np.arctan2(l2 * np.sin(joint3_alt), l1 + l2 * np.cos(joint3_alt))
    joint2_alt = alpha - beta_alt
    return (joint1, joint2, joint3)