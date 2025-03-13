import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    a = 0.425
    b = 0.39225
    c = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    cos_theta2 = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)
    cos_theta3 = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    return (theta1, theta2, theta3)