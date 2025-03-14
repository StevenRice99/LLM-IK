import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    a = 0.09465
    b = 0.0823
    r = math.sqrt(x ** 2 + y ** 2)
    if r > a + b or r < abs(a - b):
        raise ValueError('Target position is out of reach')
    alpha = math.atan2(y, x)
    theta1 = alpha + math.acos((a ** 2 + r ** 2 - b ** 2) / (2 * a * r))
    theta2 = alpha + math.acos((a ** 2 + b ** 2 - r ** 2) / (2 * a * b))
    return (theta1, theta2)