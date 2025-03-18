import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L1 = 0.09465
    L2 = 0.0823
    theta1 = math.atan2(y, x)
    theta2 = math.atan2(math.sqrt(x ** 2 + y ** 2), z - L1)
    return (theta1, theta2)