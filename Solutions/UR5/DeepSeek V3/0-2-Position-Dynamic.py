import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    cos_theta1 = math.cos(theta1)
    sin_theta1 = math.sin(theta1)
    x_rot = x * cos_theta1 + y * sin_theta1
    y_rot = -x * sin_theta1 + y * cos_theta1
    z_rot = z - 0.13585
    distance = math.sqrt(x_rot ** 2 + y_rot ** 2 + z_rot ** 2)
    a = 0.425
    b = 0.39225
    c = distance
    cos_theta3 = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3)
    sin_theta3 = math.sin(theta3)
    theta2 = math.asin(b * sin_theta3 / c)
    return (theta1, theta2, theta3)