import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    z -= 0.0823
    theta1 = np.arctan2(x, z)
    d = np.sqrt(x ** 2 + z ** 2)
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    cos_theta2 = (d ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    theta2 = np.arccos(cos_theta2)
    theta3 = np.arctan2(y, l2 + l3)
    theta4 = 0
    return (theta1, theta2, theta3, theta4)