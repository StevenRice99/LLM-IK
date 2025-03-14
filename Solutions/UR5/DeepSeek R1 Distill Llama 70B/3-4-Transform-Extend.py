import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L2 = 0.093
    L3 = 0.09465
    if x == 0:
        theta2 = math.pi / 2 if y > 0 else -math.pi / 2
    else:
        theta2 = math.atan2(y, x)
    theta1 = 0.0
    return (theta1, theta2)