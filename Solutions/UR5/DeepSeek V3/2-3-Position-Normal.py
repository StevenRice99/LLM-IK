import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    theta1 = np.arctan2(y, z)
    sin_theta1_plus_theta2 = x / L2
    cos_theta1_plus_theta2 = np.sqrt(1 - sin_theta1_plus_theta2 ** 2)
    theta1_plus_theta2 = np.arctan2(sin_theta1_plus_theta2, cos_theta1_plus_theta2)
    theta2 = theta1_plus_theta2 - theta1
    return (theta1, theta2)