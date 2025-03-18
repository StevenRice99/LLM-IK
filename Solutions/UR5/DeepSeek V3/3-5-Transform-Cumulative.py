import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    theta1 = math.atan2(px, pz)
    theta2 = math.atan2(py, math.sqrt(px ** 2 + pz ** 2))
    theta3 = rz - theta1
    return (theta1, theta2, theta3)