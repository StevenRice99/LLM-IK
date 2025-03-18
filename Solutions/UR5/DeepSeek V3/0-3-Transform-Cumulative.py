import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [rx, ry, rz].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    theta1 = math.atan2(y, x)
    x_proj = math.sqrt(x ** 2 + y ** 2)
    z_proj = z
    L2 = 0.425
    L3 = 0.39225
    D_sq = (x_proj ** 2 + z_proj ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    D_sq = max(-1, min(1, D_sq))
    theta3 = math.atan2(math.sqrt(1 - D_sq ** 2), D_sq)
    theta2 = math.atan2(z_proj, x_proj) - math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta4 = rz - (theta2 + theta3)
    return (theta1, theta2, theta3, theta4)