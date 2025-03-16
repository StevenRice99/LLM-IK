import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    l4 = 0.0823
    theta1 = math.atan2(x, z)
    r_xz = math.sqrt(x * x + z * z)
    theta3 = math.atan2(x, z)
    wrist_x = x - l4 * math.sin(theta3)
    wrist_y = y - l4 * math.cos(theta3)
    wrist_z = z
    h = wrist_z - l1
    y_eff = wrist_y - l2
    d = math.sqrt(h * h + y_eff * y_eff)
    cos_theta4 = (l3 * l3 + l3 * l3 - d * d) / (2 * l3 * l3)
    cos_theta4 = max(min(cos_theta4, 1.0), -1.0)
    theta4 = math.acos(cos_theta4)
    alpha = math.atan2(y_eff, h)
    beta = math.asin(l3 * math.sin(theta4) / d)
    theta2 = alpha - beta
    return (theta1, theta2, theta3, theta4)