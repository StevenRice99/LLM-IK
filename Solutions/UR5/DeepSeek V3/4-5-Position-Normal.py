import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.09465
    L2 = 0.0823
    z_over_L2 = z / L2
    z_over_L2 = max(-1.0, min(1.0, z_over_L2))
    theta2 = math.acos(z_over_L2)
    if abs(theta2) < 1e-10 or abs(theta2 - math.pi) < 1e-10:
        theta1 = math.atan2(y - L1, x)
    else:
        sin_theta2 = math.sin(theta2)
        theta1 = math.atan2((y - L1) / (L2 * sin_theta2), x / (L2 * sin_theta2))
    return (theta1, theta2)