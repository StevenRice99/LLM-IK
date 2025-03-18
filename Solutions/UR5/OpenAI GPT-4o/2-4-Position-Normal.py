import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    z_joint3 = z - 0.09465
    y_joint3 = y - 0.093
    theta1 = np.arctan2(x, z_joint3)
    r = np.sqrt(x ** 2 + z_joint3 ** 2)
    d = 0.39225
    theta2 = np.arcsin((z_joint3 - d) / r)
    theta3 = 0
    return (theta1, theta2, theta3)