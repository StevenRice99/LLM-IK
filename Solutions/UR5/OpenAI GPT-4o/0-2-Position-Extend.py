import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    tcp_offset = 0.39225
    z_adjusted = z - tcp_offset
    ratio = z_adjusted / 0.425
    ratio = max(-1, min(1, ratio))
    theta2 = math.acos(ratio)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    K = 0.425 * sin_theta2
    L = 0.01615
    denominator = K ** 2 + L ** 2
    cos_theta1 = (K * x + L * y) / denominator
    sin_theta1 = (-L * x + K * y) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    z_contribution = 0.425 * cos_theta2
    z_remaining = z_adjusted - z_contribution
    ratio3 = z_remaining / 0.425
    ratio3 = max(-1, min(1, ratio3))
    theta3 = math.acos(ratio3)
    theta3 = -theta3
    return (theta1, theta2, theta3)