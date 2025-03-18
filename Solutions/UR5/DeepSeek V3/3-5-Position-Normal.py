import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p].
    """
    x, y, z = p
    l1 = 0.093
    l2 = 0.09465
    l_tcp = 0.0823
    theta2 = np.arcsin((y - l_tcp) / (l1 + l2))
    theta1 = np.arctan2(x, z)
    x_prime = x - l1 * np.cos(theta1)
    z_prime = z - l1 * np.sin(theta1)
    theta3 = np.arctan2(x_prime, z_prime)
    return (theta1, theta2, theta3)