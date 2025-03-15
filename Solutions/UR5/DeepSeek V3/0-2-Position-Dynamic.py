import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    target_x = x
    target_y = y
    target_z = z - L1
    d = math.sqrt(target_x ** 2 + target_y ** 2)
    theta1 = math.atan2(target_y, target_x)
    D = math.sqrt(d ** 2 + (target_z - L1) ** 2)
    cos_theta2 = (L2 ** 2 + D ** 2 - L3 ** 2) / (2 * L2 * D)
    cos_theta2 = max(min(cos_theta2, 1), -1)
    theta2 = math.acos(cos_theta2) - math.atan2(target_z - L1, d)
    cos_theta3 = (L2 ** 2 + L3 ** 2 - D ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = math.acos(cos_theta3) - math.pi
    return (theta1, theta2, theta3)