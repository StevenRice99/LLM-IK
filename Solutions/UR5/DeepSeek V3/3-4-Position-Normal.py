import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.093
    L2 = 0.09465
    cos_theta1 = (y - L1) / L2
    theta1 = math.acos(cos_theta1)
    theta2 = math.atan2(z, x)
    return (theta1, theta2)