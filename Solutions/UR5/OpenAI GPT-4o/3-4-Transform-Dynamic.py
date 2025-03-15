import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    L1 = 0.093
    L2 = 0.09465
    theta1 = math.atan2(py, px)
    r = math.sqrt(px ** 2 + py ** 2)
    sin_theta2 = (pz - L1) / L2
    sin_theta2 = max(-1.0, min(1.0, sin_theta2))
    theta2 = math.asin(sin_theta2)
    return (theta1, theta2)