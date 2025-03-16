import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.09465
    L2 = 0.0823
    theta1 = math.atan2(-x, y)
    theta2 = 0
    return (theta1, theta2)