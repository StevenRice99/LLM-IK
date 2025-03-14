import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.093
    L2 = 0.09465
    theta1 = math.asin(z / L1)
    theta2 = math.asin(-x / L2)
    return (theta1, theta2)