import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    x_prime = x * math.cos(theta1) + z * math.sin(theta1)
    y_prime = y - 0.093
    z_prime = -x * math.sin(theta1) + z * math.cos(theta1)
    d = math.sqrt(x_prime ** 2 + y_prime ** 2)
    theta2 = math.atan2(y_prime, x_prime) - math.atan2(0.0823, d)
    theta3 = math.atan2(0.0823, d)
    return (theta1, theta2, theta3)