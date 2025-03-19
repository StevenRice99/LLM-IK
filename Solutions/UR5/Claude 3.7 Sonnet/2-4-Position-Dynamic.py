import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    theta1 = math.atan2(x, z)
    r = math.sqrt(x ** 2 + z ** 2)
    r_adjusted = r - l2
    height_diff = y - l3
    theta2 = math.atan2(height_diff, r_adjusted)
    theta3 = 0.0
    return (theta1, theta2, theta3)