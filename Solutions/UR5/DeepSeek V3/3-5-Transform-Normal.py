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
    theta1 = np.arctan2(y, x)
    theta2 = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    theta3 = yaw - theta1
    return (theta1, theta2, theta3)