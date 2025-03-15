import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(x, z)
    theta2 = 0.0
    remaining_distance = math.sqrt(x ** 2 + y ** 2 + z ** 2) - (0.1197 + 0.425 + 0.39225)
    theta3 = math.atan2(remaining_distance, 0.093)
    return (theta1, theta2, theta3)