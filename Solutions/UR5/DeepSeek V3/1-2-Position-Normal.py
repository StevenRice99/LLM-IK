import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    d = 0.1197
    r = math.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta2_pos = math.acos(cos_theta2)
    theta2_neg = -theta2_pos
    alpha = math.atan2(z, x)
    beta_pos = math.acos((L1 ** 2 + r ** 2 - L2 ** 2) / (2 * L1 * r))
    theta1_pos = alpha - beta_pos
    theta1_neg = alpha + beta_pos

    def forward_kinematics(theta1, theta2):
        x_pos = L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2)
        z_pos = L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2)
        return (x_pos, z_pos)
    x_pos_pos, z_pos_pos = forward_kinematics(theta1_pos, theta2_pos)
    x_pos_neg, z_pos_neg = forward_kinematics(theta1_neg, theta2_neg)
    error_pos = (x_pos_pos - x) ** 2 + (z_pos_pos - z) ** 2
    error_neg = (x_pos_neg - x) ** 2 + (z_pos_neg - z) ** 2
    if error_pos < error_neg:
        return (theta1_pos, theta2_pos)
    else:
        return (theta1_neg, theta2_neg)