import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    px, py, pz = p
    theta1 = math.atan2(px, pz)
    theta2 = 0.0
    return (theta1, theta2)