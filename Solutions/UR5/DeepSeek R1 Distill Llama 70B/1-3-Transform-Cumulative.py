import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    theta1 = math.atan2(p[0], p[2])
    adjusted_x = p[0] * math.cos(theta1) + p[2] * math.sin(theta1)
    adjusted_z = -p[0] * math.sin(theta1) + p[2] * math.cos(theta1)
    theta2 = math.atan2(adjusted_x, adjusted_z)
    theta3 = r[1]
    return (theta1, theta2, theta3)