import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    theta2 = 0.0
    theta1 = math.atan2(px, pz)
    return (theta1, theta2)