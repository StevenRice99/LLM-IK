import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x_target, y_target, z_target = p
    z_ratio = z_target / 0.425
    if z_ratio < -1.0:
        z_ratio = -1.0
    elif z_ratio > 1.0:
        z_ratio = 1.0
    theta2 = math.acos(z_ratio)
    y_adjusted = y_target + 0.1197
    y_ratio = y_adjusted / 0.13585
    if y_ratio < -1.0:
        y_ratio = -1.0
    elif y_ratio > 1.0:
        y_ratio = 1.0
    theta1 = math.acos(y_ratio)
    return (theta1, theta2)