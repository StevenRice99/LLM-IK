import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    z_normalized = z / 0.00165
    z_clamped = max(min(z_normalized, 1.0), -1.0)
    theta1 = math.acos(z_clamped)
    S = math.sin(theta1)
    A = 0.093
    B = 0.09465
    C = (x - S * B) / A
    D = (y - B) / A
    denominator = S ** 2 + 1
    cos_theta2 = (D + S * C) / denominator
    sin_theta2 = (S * D - C) / denominator
    theta2 = math.atan2(sin_theta2, cos_theta2)
    return (theta1, theta2)