import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    sin_phi = -px / 0.1197
    cos_phi = (py - 0.093) / 0.1197
    magnitude = math.sqrt(sin_phi ** 2 + cos_phi ** 2)
    if magnitude > 1:
        sin_phi /= magnitude
        cos_phi /= magnitude
    phi = math.atan2(sin_phi, cos_phi)
    theta1 = phi
    theta2 = 0.0
    theta3 = 0.0
    return (theta1, theta2, theta3)