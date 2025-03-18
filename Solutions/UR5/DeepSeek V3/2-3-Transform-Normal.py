import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    phi, theta, psi = r
    theta1 = np.arctan2(x, z)
    theta2 = theta - theta1
    theta2 = np.mod(theta2 + np.pi, 2 * np.pi) - np.pi
    return (theta1, theta2)