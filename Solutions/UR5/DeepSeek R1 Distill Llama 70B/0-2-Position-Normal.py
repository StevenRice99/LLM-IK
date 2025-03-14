import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    a = 0.13585
    b = 0.425
    c = 0.1197
    d = 0.39225
    try:
        cos_theta1 = (y + c) / a
        if cos_theta1 < -1.0 or cos_theta1 > 1.0:
            raise ValueError('The target position is not reachable with the given configuration.')
        theta1 = math.acos(cos_theta1)
        sin_theta1 = math.sqrt(1 - cos_theta1 ** 2)
        cos_theta2 = (z - d) / b
        if cos_theta2 < -1.0 or cos_theta2 > 1.0:
            raise ValueError('The target position is not reachable with the given configuration.')
        theta2 = math.acos(cos_theta2)
        sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
        x_calculated = a * sin_theta1 + b * sin_theta2
        if not math.isclose(x_calculated, x, rel_tol=1e-09):
            raise ValueError('The target position is not reachable with the given configuration.')
        theta3 = 0.0
        return (theta1, theta2, theta3)
    except ValueError as e:
        pass
        return (0.0, 0.0, 0.0)