import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    l1 = 0.093
    l2 = 0.09465
    l3 = 0.0823
    theta1 = math.atan2(px, pz)
    r = math.sqrt(px ** 2 + pz ** 2)
    d_squared = r ** 2 + (py - l1) ** 2
    cos_theta3 = (d_squared - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3 = math.atan2(math.sqrt(1 - cos_theta3 ** 2), cos_theta3)
    alpha = math.atan2(py - l1, r)
    beta = math.atan2(l3 * math.sin(theta3), l2 + l3 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)