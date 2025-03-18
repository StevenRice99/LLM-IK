import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    r = math.sqrt(x ** 2 + z ** 2)
    theta1 = math.atan2(x, z)
    d = 0.39225
    adjusted_r = r - d
    theta2 = math.atan2(adjusted_r, z) - theta1
    theta3 = 0
    return (theta1, theta2, theta3)