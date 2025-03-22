import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.13585
    l2 = 0.425
    l3 = 0.39225
    l4 = 0.093
    l5 = 0.09465
    q1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    joint5_x = x - l5 * np.sin(q1) * np.sin(0)
    joint5_y = y + l5 * np.cos(q1) * np.sin(0)
    joint5_z = z - l5 * np.cos(0)
    joint5_r = np.sqrt(joint5_x ** 2 + joint5_y ** 2)
    joint5_r_adj = np.sqrt((joint5_r - l4) ** 2 + (joint5_z - l1) ** 2)
    cos_q3 = (joint5_r_adj ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    beta = np.arctan2(joint5_z - l1, joint5_r - l4)
    alpha = np.arctan2(l3 * np.sin(q3), l2 + l3 * np.cos(q3))
    q2 = beta - alpha
    q4 = -q2 - q3
    q5 = 0.0
    return (q1, q2, q3, q4, q5)