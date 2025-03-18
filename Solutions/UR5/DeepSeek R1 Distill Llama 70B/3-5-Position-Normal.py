import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    numerator_y = y - 0.0823
    denominator = 0.093
    sin_phi = numerator_y / denominator
    A = math.sqrt(x ** 2 + z ** 2)
    cos_phi = (A - 0.09465) / 0.093
    phi = math.atan2(sin_phi, cos_phi)
    cos_theta3 = x / A
    sin_theta3 = -z / A
    theta3 = math.atan2(sin_theta3, cos_theta3)
    theta1 = 0.0
    theta2 = phi - theta1
    return (theta1, theta2, theta3)