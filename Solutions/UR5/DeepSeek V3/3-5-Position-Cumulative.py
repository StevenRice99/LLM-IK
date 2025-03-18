import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    theta2 = math.atan2(y, math.sqrt(x ** 2 + z ** 2) - 0.093)
    theta3 = math.atan2(y, math.sqrt(x ** 2 + z ** 2) - 0.093 - 0.09465)
    return (theta1, theta2, theta3)