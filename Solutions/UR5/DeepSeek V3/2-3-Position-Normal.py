import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    if abs(z / L2) > 1:
        raise ValueError('The target position is not reachable.')
    theta2 = math.asin(z / L2)
    denominator = L1 + L2 * math.cos(theta2)
    theta1 = math.atan2(y / denominator, x / denominator)
    return (theta1, theta2)