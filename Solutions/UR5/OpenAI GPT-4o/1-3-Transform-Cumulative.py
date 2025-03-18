import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    adjusted_x = x
    adjusted_y = y + 0.1197
    adjusted_z = z - 0.425
    theta2 = math.atan2(adjusted_x, adjusted_z)
    adjusted_z -= 0.39225
    theta3 = math.atan2(adjusted_x, adjusted_z)
    return (theta1, theta2, theta3)