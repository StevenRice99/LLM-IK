import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    z_offset = z - 0.09465
    theta2 = math.atan2(z_offset, math.sqrt(x ** 2 + y ** 2))
    return (theta1, theta2)