import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    joint1 = math.atan2(x, z)
    y_proj = y
    z_proj = math.sqrt(x ** 2 + z ** 2) - L3
    D = math.sqrt(y_proj ** 2 + z_proj ** 2)
    if D > L1 + L2 or D < abs(L1 - L2):
        raise ValueError('Target position is not reachable')
    cos_joint3 = (L1 ** 2 + L2 ** 2 - D ** 2) / (2 * L1 * L2)
    cos_joint3 = max(min(cos_joint3, 1), -1)
    joint3 = math.acos(cos_joint3)
    alpha = math.atan2(z_proj, y_proj)
    beta = math.atan2(L2 * math.sin(joint3), L1 + L2 * math.cos(joint3))
    joint2 = alpha - beta
    return (joint1, joint2, joint3)