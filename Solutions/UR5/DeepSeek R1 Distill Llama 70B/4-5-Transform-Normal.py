import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    cos_theta2 = (pz - 0.09465) / 0.0823
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(py, px)
    return (theta1, theta2)