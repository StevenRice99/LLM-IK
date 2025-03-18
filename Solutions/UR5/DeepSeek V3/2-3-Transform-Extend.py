import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    D = math.sqrt(x ** 2 + z ** 2)
    if D > L1 + L2 or D < abs(L1 - L2):
        raise ValueError('Target position is not reachable')
    cos_theta2 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta2 = math.acos(cos_theta2)
    theta2_options = [theta2, -theta2]
    theta1_options = []
    for theta2_option in theta2_options:
        alpha = math.atan2(z, x)
        beta = math.atan2(L2 * math.sin(theta2_option), L1 + L2 * math.cos(theta2_option))
        theta1_option = alpha - beta
        theta1_options.append(theta1_option)
    desired_orientation = r[1]
    best_solution = None
    min_error = float('inf')
    for theta1, theta2 in zip(theta1_options, theta2_options):
        current_orientation = theta1 + theta2
        error = abs(current_orientation - desired_orientation)
        if error < min_error:
            min_error = error
            best_solution = (theta1, theta2)
    return best_solution