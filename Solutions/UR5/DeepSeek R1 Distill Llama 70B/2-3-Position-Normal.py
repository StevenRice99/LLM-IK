import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    alpha = math.atan2(x, y)
    sin_theta2 = y / L2
    theta2 = math.asin(sin_theta2)
    cos_theta1 = x / L1
    theta1 = math.acos(cos_theta1)
    z_calculated = L1 * math.cos(theta1) + L2 * math.sin(theta2)
    if not math.isclose(z, z_calculated, rel_tol=1e-09):
        theta1 = -theta1
        z_calculated = L1 * math.cos(theta1) + L2 * math.sin(theta2)
    return (theta1, theta2)