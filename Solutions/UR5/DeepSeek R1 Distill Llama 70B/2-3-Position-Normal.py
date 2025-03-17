import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    p_x, _, p_z = p
    A = 0.39225
    B = 0.093
    M = math.sqrt(p_x ** 2 + p_z ** 2)
    if M == 0:
        theta1 = 0.0
        theta2 = 0.0
        return (theta1, theta2)
    phi = math.atan2(p_z, p_x)
    K = (A ** 2 + B ** 2 - M ** 2) / (2 * A * B)
    K = max(min(K, 1.0), -1.0)
    theta1_minus_phi = math.acos(K)
    theta1 = phi + theta1_minus_phi
    sin_theta2 = A * math.sin(theta1_minus_phi) / B
    sin_theta2 = max(min(sin_theta2, 1.0), -1.0)
    cos_theta2 = math.sqrt(max(0, 1 - sin_theta2 ** 2))
    theta2 = math.atan2(sin_theta2, cos_theta2)
    return (theta1, theta2)