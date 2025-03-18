import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    p2_x = 0.425 * math.sin(theta1)
    p2_z = 0.425 * math.cos(theta1)
    v_x = x - p2_x
    v_z = z - p2_z
    theta2 = math.atan2(v_x, v_z)
    return (theta1, theta2)