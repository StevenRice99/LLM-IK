import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.425
    B = 0.39225
    D = x ** 2 + z ** 2 + A ** 2 - B ** 2
    P = D / (2 * A)
    sqrt_xz = math.sqrt(x ** 2 + z ** 2)
    if x == 0:
        alpha = math.pi / 2
    else:
        alpha = math.atan(z / x)
    ratio = P / sqrt_xz
    ratio = max(min(ratio, 1.0), -1.0)
    theta1_sol1 = math.asin(ratio) - alpha
    theta1_sol2 = math.pi - math.asin(ratio) - alpha
    arg1 = (z - A * math.cos(theta1_sol1)) / B
    arg1 = max(min(arg1, 1.0), -1.0)
    theta2_sol1 = math.acos(arg1)
    arg2 = (z - A * math.cos(theta1_sol2)) / B
    arg2 = max(min(arg2, 1.0), -1.0)
    theta2_sol2 = math.acos(arg2)
    return (theta1_sol1, theta2_sol1)