import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    px = x - 0.39225 * math.sin(theta1)
    pz = z - 0.39225 * math.cos(theta1)
    theta2 = math.atan2(px, pz)
    py = y - 0.093
    theta3 = math.atan2(py, math.sqrt(px ** 2 + pz ** 2))
    px = x - 0.09465 * math.sin(theta1 + theta2 + theta3)
    pz = z - 0.09465 * math.cos(theta1 + theta2 + theta3)
    theta4 = math.atan2(px, pz)
    return (theta1, theta2, theta3, theta4)