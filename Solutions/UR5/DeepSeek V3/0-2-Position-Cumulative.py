import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(-x, y)
    z_normalized = z / 0.425
    if z_normalized < -1:
        z_normalized = -1
    elif z_normalized > 1:
        z_normalized = 1
    theta2 = math.acos(z_normalized)
    theta3 = math.atan2(x, z)
    return (theta1, theta2, theta3)